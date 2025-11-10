pub mod models;

use burn::backend::Wgpu;
use burn_ndarray::NdArray;

use crate::models::stardist::Model;
use crate::models::test_model::ModelConfig;

type Backend = Wgpu<f32, i32>;

fn main() {
    // load a pretrained model
    let stardist_model: Model<NdArray<f32>> = Model::default();
    println!("[DEBUG] fixed stardist model info: \n{}", stardist_model);

    // create a test custom model
    let device = Default::default();
    let custom_model = ModelConfig::new(10, 512).init::<Backend>(&device);
    println!("[DEBUG] custom model info: \n{}", custom_model);
}
