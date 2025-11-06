mod model;

use burn::prelude::*;
use burn::backend::Wgpu;

use crate::model::ModelConfig;

type Backend = Wgpu<f32, i32>;

fn main() {
    let device = Default::default();
    let model = ModelConfig::new(10, 512).init::<Backend>(&device);

    println!("[DEBUG] model info: {}", model);
}
