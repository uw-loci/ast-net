use imgal::prelude::*;
use ndarray::ArrayView3;

/// TODO
pub fn batch_calibrated_gs<T>(
    batch_data: Vec<ArrayView3<T>>,
    calibration_data: ArrayView3<T>,
) -> Result<(), ImgalError>
where
    T: AsNumeric,
{
    todo!();
}
