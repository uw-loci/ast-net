pub mod stardist {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/pretrained_models/stardist_2d_versatile_fluo_fixed.rs"
    ));
}
pub mod test_model;
