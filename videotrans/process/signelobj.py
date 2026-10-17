import multiprocessing, os
import time
import tempfile
from pathlib import Path

from videotrans.configure.config import app_cfg, settings, logger

def pool_init_worker():
    import sys
    class NullWriter:
        def write(self, *args, **kwargs): pass
        def flush(self, *args, **kwargs): pass
        @staticmethod
        def isatty(): return False
        @staticmethod
        def fileno(): return -1
        @staticmethod
        def read(*args, **kwargs): return ""
        def close(self): pass
        @property
        def encoding(self): return "utf-8"
        @property
        def closed(self): return False

    _null_writer = NullWriter()
    for _stream in ['stdout', 'stderr', 'stdin', '__stdout__', '__stderr__', '__stdin__']:
        if getattr(sys, _stream, None) is None:
            setattr(sys, _stream, _null_writer)


def _task_worker_wrapper(func, kwargs, marker):
    if not Path(marker).exists():
        return None, "Task cancelled before execution"
    Path(marker).write_text(str(os.getpid()), encoding="ascii")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # 这里执行真正的任务
    return func(**kwargs)


class AsyncResultFutureWrapper:
    """Track the actual worker, not just replacement workers in the pool."""
    def __init__(self, async_result, pool_executor, marker, progress_file=None, idle_timeout=7200):
        self.async_result = async_result
        self._pool = pool_executor
        self.marker = Path(marker)
        self.progress_file = Path(progress_file) if progress_file else None
        self.idle_timeout = idle_timeout
        self.last_progress = time.monotonic()
        self.progress_stamp = None
        self.worker_pid = None
        self.error = None

    def _fail(self, message):
        self.error = message
        # Terminate only this task's worker, never other jobs sharing the pool.
        for worker in getattr(self._pool, '_pool', []):
            if worker.pid == self.worker_pid and worker.is_alive():
                worker.terminate()
                break
        self.marker.unlink(missing_ok=True)

    def done(self):
        if self.error:
            return True
        if self.async_result.ready():
            self.marker.unlink(missing_ok=True)
            return True
        if self.worker_pid is None:
            try:
                self.worker_pid = int(self.marker.read_text())
                self.last_progress = time.monotonic()
            except (OSError, ValueError):
                pass
        if self.worker_pid is not None:
            workers = getattr(self._pool, '_pool', [])
            if not any(w.pid == self.worker_pid and w.is_alive() for w in workers):
                self._fail('Subprocess exited before returning its result (possible crash/OOM)')
                return True
        if self.progress_file:
            try:
                stamp = self.progress_file.stat().st_mtime_ns
                if stamp != self.progress_stamp:
                    self.progress_stamp = stamp
                    self.last_progress = time.monotonic()
            except OSError:
                pass
        if time.monotonic() - self.last_progress > self.idle_timeout:
            self._fail(f'Subprocess made no progress for {self.idle_timeout}s')
            return True
        return False

    def cancel(self):
        self._fail("Task interrupted by user")

    def result(self, timeout=None):
        started = time.monotonic()
        while not self.done():
            if app_cfg.exit_soft:
                self._fail('Task interrupted by user')
                break
            if timeout is not None and time.monotonic()-started > timeout:
                self._fail('Task wait timed out')
                break
            time.sleep(0.1)
        if self.error:
            return False, self.error
        try:
            return self.async_result.get(timeout=1)
        except Exception as error:
            return False, f'Subprocess Error: {error}'


# ==========================================
# 全局单例管理器
# ==========================================

class GlobalProcessManager:
    _executor_cpu = None
    _executor_gpu = None



    @classmethod
    def get_gpu_process_nums(cls):
        cpu_count = max(int(os.cpu_count()),1)
        try:
            process_max_gpu = max( int(float(settings.get('process_max_gpu', 0))) ,1)
        except (TypeError, ValueError):
            process_max_gpu = 1

        return int(min(process_max_gpu, 8, cpu_count))


    @classmethod
    def get_executor_cpu(cls):
        if cls._executor_cpu is None:
            ctx = multiprocessing.get_context('spawn')
            max_workers = app_cfg.MAX_CPU_PROCESS
            logger.debug(f'CPU进程池:{max_workers=}')
            cls._executor_cpu = ctx.Pool(
                processes=int(max_workers),
                initializer=pool_init_worker, 
                maxtasksperchild=1  # <--- CPU 也让它跑完就死，彻底释放物理内存
            )
        return cls._executor_cpu

    @classmethod
    def get_executor_gpu(cls):
        """
        max_workers 设为 1，意味着同一时间只能跑一个 AI 任务。
        """
        if cls._executor_gpu is None:
            ctx = multiprocessing.get_context('spawn')
            max_workers = app_cfg.MAX_GPU_PROCESS
            logger.debug(f'GPU进程池:{max_workers=}')
            cls._executor_gpu = ctx.Pool(
                processes=int(max_workers),
                initializer=pool_init_worker, 
                maxtasksperchild=1
            )

        return cls._executor_gpu

    @classmethod
    def _submit(cls, executor, func, kwargs):
        from videotrans.configure.config import TEMP_ROOT
        folder = Path(TEMP_ROOT, 'process_tasks')
        folder.mkdir(parents=True, exist_ok=True)
        fd, marker = tempfile.mkstemp(dir=folder, suffix='.pid')
        os.close(fd)
        # Separation emits progress per bounded chunk. Other models retain a
        # generous limit until they expose comparable progress events.
        separation = func.__name__ in ('vocal_bgm', 'vocal_bgm_spleeter')
        if separation and not kwargs.get('logs_file'):
            kwargs['logs_file'] = marker + '.progress'
        try:
            result = executor.apply_async(_task_worker_wrapper, args=(func, kwargs, marker))
        except Exception:
            Path(marker).unlink(missing_ok=True)
            raise
        return AsyncResultFutureWrapper(result, executor, marker,
            progress_file=kwargs.get('logs_file'), idle_timeout=900 if separation else 7200)

    @classmethod
    def submit_task_cpu(cls, func, **kwargs):
        return cls._submit(cls.get_executor_cpu(), func, kwargs)

    @classmethod
    def submit_task_gpu(cls, func, **kwargs):
        return cls._submit(cls.get_executor_gpu(), func, kwargs)

    @classmethod
    def shutdown(cls):
        if cls._executor_cpu:
            cls._executor_cpu.terminate()
            cls._executor_cpu.join()
            cls._executor_cpu = None
        if cls._executor_gpu:
            cls._executor_gpu.terminate()
            cls._executor_gpu.join()
            cls._executor_gpu = None
