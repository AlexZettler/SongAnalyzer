from song_analyzer.schema import AnalysisResult, NoteEvent, StemAudioRef


def test_analysis_result_json_roundtrip():
    r = AnalysisResult(
        source_path="x.wav",
        sample_rate=44100,
        duration_s=1.0,
        stems=[StemAudioRef(stem_id="other", path=None)],
        notes=[
            NoteEvent(
                start_time_s=0.0,
                end_time_s=0.1,
                midi_pitch=60,
                velocity=0.9,
                stem_id="other",
            )
        ],
    )
    s = r.model_dump_json(indent=2)
    r2 = AnalysisResult.model_validate_json(s)
    assert r2.source_path == r.source_path
    assert r2.notes[0].midi_pitch == 60
