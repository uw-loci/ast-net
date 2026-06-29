use imgal::prelude::*;
use ndarray::{Array3, ArrayView3};

/// TODO
pub fn batch_calibrated_gs<T>(
    batch_data: &Vec<ArrayView3<T>>,
    calibration_data: ArrayView3<T>,
) -> Result<Vec<Array3<f64>>, ImgalError>
where
    T: AsNumeric,
{
    todo!();
}
