from types import SimpleNamespace
from videotrans.process import signelobj as mod


def test_replacement_worker_does_not_hide_lost_task(tmp_path):
    marker = tmp_path/'pid'; marker.write_text('42')
    result = SimpleNamespace(ready=lambda:False)
    pool = SimpleNamespace(_pool=[SimpleNamespace(pid=43,is_alive=lambda:True)])
    future = mod.AsyncResultFutureWrapper(result,pool,marker)
    assert future.done()
    assert 'exited' in future.result()[1]


def test_stalled_worker_terminated_without_other_jobs(tmp_path,monkeypatch):
    clock=[0]
    monkeypatch.setattr(mod.time,'monotonic',lambda:clock[0])
    marker=tmp_path/'pid';marker.write_text('42')
    killed=[]
    workers=[SimpleNamespace(pid=p,is_alive=lambda:True,terminate=lambda p=p:killed.append(p)) for p in (42,43)]
    future=mod.AsyncResultFutureWrapper(SimpleNamespace(ready=lambda:False),SimpleNamespace(_pool=workers),marker,idle_timeout=10)
    assert not future.done()
    clock[0]=11
    assert future.done()
    assert killed==[42]
    assert 'no progress' in future.result()[1]


def test_progress_extends_idle_deadline(tmp_path,monkeypatch):
    clock=[0]
    monkeypatch.setattr(mod.time,'monotonic',lambda:clock[0])
    marker=tmp_path/'pid';marker.write_text('42')
    progress=tmp_path/'progress';progress.write_text('1')
    future=mod.AsyncResultFutureWrapper(SimpleNamespace(ready=lambda:False),
        SimpleNamespace(_pool=[SimpleNamespace(pid=42,is_alive=lambda:True)]),marker,progress,10)
    assert not future.done()
    clock[0]=9;progress.write_text('2')
    assert not future.done()
    clock[0]=18
    assert not future.done()


def _crash_worker():
    import os
    os._exit(7)


def test_real_pool_worker_crash_finishes_future(tmp_path):
    import multiprocessing
    import time
    pool=multiprocessing.get_context('spawn').Pool(1)
    marker=tmp_path/'pid';marker.touch()
    try:
        result=pool.apply_async(mod._task_worker_wrapper,(_crash_worker,{},str(marker)))
        future=mod.AsyncResultFutureWrapper(result,pool,marker,idle_timeout=10)
        deadline=time.monotonic()+10
        while not future.done() and time.monotonic()<deadline:
            time.sleep(.05)
        assert future.done()
        assert 'exited' in future.result()[1]
    finally:
        pool.terminate()
        pool.join()
