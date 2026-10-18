from videotrans.util._render_progress import render_progress, progress_message


def test_eta_from_encoded_time_and_speed():
    assert render_progress(['out_time_us=30000000','speed=2.0x'],120000)==(25,45)
    assert '25.0%' in progress_message('Render',['out_time_us=30000000','speed=2x'],120000)


def test_invalid_and_startup_speed_do_not_invent_eta():
    assert render_progress(['out_time_us=1000000','speed=2x'],120000)==(100/120,None)
    assert render_progress(['out_time_us=30000000','speed=N/A'],120000) is None
    assert render_progress(['out_time_us=30000000','speed=0x'],120000)==(25,None)
    assert 'finalizing' in progress_message('Render',['out_time_us=120000000','speed=2x'],120000)
