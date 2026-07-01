use std::collections::HashMap;

use cellcast::models::StarDist2D;
use imgal::phasor::time_domain::gs_roi;
use imgal::prelude::*;
use imgal::spatial::roi::roi_cloud_map;
use imgal::transform::project::sum_project;
use ndarray::{Array2, Array3, ArrayView3};

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
    let batch_roi_gs: Vec<HashMap<u64, Array2<f64>>> = batch_data
        .iter()
        .zip(batch_roi_clouds.iter())
        .map(|(d, c)| gs_roi(d, period, c, None, None, Some(1)).unwrap())
        .collect();
    Ok(batch_roi_gs)
}
