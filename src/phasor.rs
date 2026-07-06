use std::collections::HashMap;

use cellcast::models::StarDist2D;
use imgal::parameter::omega;
use imgal::phasor::calibration::{calibrate_gs_image_mut, modulation_and_phase};
use imgal::phasor::time_domain::gs_roi;
use imgal::prelude::*;
use imgal::spatial::roi::roi_cloud_map;
use imgal::threshold::global::otsu_mask;
use imgal::transform::project::sum_project;
use ndarray::{Array2, ArrayView3, Axis};

/// Batch process a set of FLIM data.
///
/// # Description
///
/// Batch processes a set of FLIM data by segmenting each array with StarDist2D
/// and computing the calibrated G/S values for each ROI.
///
/// # Arguments
///
/// * `batch_data`:
/// * `calibration_data`:
/// * `period`:
///
/// # Returns
///
/// * `Ok(Vec<Array2<f64>>)`: A vec of G, S coordinate point clouds.
/// * `Err(ImgalError)`:
pub fn batch_segment_gs(
    batch_data: &Vec<ArrayView3<u16>>,
    calibration_data: ArrayView3<u16>,
    calibration_tau: f64,
    period: f64,
) -> Result<Vec<HashMap<u64, Array2<f64>>>, ImgalError> {
    // TODO expose `axis` param
    // calculate batch per roi G/S values
    let sd = StarDist2D::init_fluo(None, true).expect("Failed to initialize StarDist2D.");
    let batch_labels: Vec<Array2<u64>> = batch_data
        .iter()
        .map(|&v| {
            let arr_sum = sum_project(v, Some(2), Some(1)).unwrap();
            sd.predict_fluo(&arr_sum, None, None, None, None).unwrap()
        })
        .collect();
    let batch_roi_clouds: Vec<HashMap<u64, Array2<usize>>> = batch_labels
        .iter()
        .map(|v| roi_cloud_map(v, Some(1)))
        .collect();
    let mut batch_roi_gs: Vec<HashMap<u64, Array2<f64>>> = batch_data
        .iter()
        .zip(batch_roi_clouds.iter())
        .map(|(d, c)| gs_roi(d, period, c, None, None, Some(1)).unwrap())
        .collect();
    let mod_phs = calibration_to_mod_phs(calibration_data, calibration_tau, period);
    // TODO the code doesn't work yet, I need to write `calibrate_gs_roi_mut` in imgal
    // batch_roi_gs.iter().for_each(|v| {
    //     v.iter().for_each(|(_, v)| calibrate_gs_roi_mut(v.view_mut(), mod_phs.0, mod_phs.1, Some(Axis(2)), Some(1)))
    // });
    Ok(batch_roi_gs)
}

/// Convert calibration data (*i.e* chroma slide) to modulation and phase.
fn calibration_to_mod_phs(
    calibration_data: ArrayView3<u16>,
    calibration_tau: f64,
    period: f64,
) -> (f64, f64) {
    let arr_sum = sum_project(calibration_data, Some(2), Some(1)).unwrap();
    let mask = otsu_mask(&arr_sum, None, Some(1)).unwrap();
    let mask = mask.mapv(|v| v as u64);
    let rcm = roi_cloud_map(&mask, Some(1));
    let cal_gs = gs_roi(calibration_data, period, &rcm, None, Some(2), Some(1)).unwrap();
    let cal_gs = cal_gs.get(&1).unwrap();
    let mean_gs = cal_gs
        .mean_axis(Axis(0))
        .expect("Unable to compute mean G/S values.");
    modulation_and_phase(mean_gs[0], mean_gs[1], calibration_tau, omega(period))
}
