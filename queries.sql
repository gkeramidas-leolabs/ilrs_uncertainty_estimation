\copy (select tmq.range_res, tmq.range_md, tmq.doppler_res, tmq.doppler_md, tmq.propagation_time, tmq.state_id, tmq.tracklet_id, tmt.instrument_id, tmt.fit_epoch, tmt.fit_corrected_range, tmt.fit_corrected_doppler, tts.timestamp from tracker_measurements_quality as tmq join tracker_measurement_tracklets as tmt on tmq.tracklet_id=tmt.id join tracker_targetstate as tts on tmq.state_id=tts.id where tmq.created_at >= '2023-05-15 00:00:00'::timestamptz and tmq.created_at < '2023-05-16 00:00:00'::timestamptz) to '2023-05-15-residuals.csv' with (format csv, header)
