class Config:
    # Data paths
    data_dir = "data/refined-set"  # Directory containing raw PDB/SDF files
    affinity_file = "data/INDEX_refined_data.2020"  # Binding affinity labels file
    
    # Note: Processed dataset will be saved to ./dataset/
    # Note: Trained models will be saved to ./checkpoint/
    
    # Dataset fraction control
    # Examples:
    # dataset_fraction = 1.0   # Use full dataset (default)
    # dataset_fraction = 0.1   # Use 10% of dataset
    dataset_fraction = 0.5  # Use 1.0 for full dataset, since we already reduced to 500 samples
    
    val_split = 0.2  # 20% for validation
    test_split = 0.2  # 20% for test (from validation set)
    train_batch_size = 32  # Reduced for smaller dataset
    val_batch_size = 50   # Reduced for smaller dataset
    test_batch_size = 50  # Reduced for smaller dataset
    learning_rate = 0.065
    use_scheduler = False
    step_size = 5
    gamma = 0.25
    in_channels = 5
    num_gnn_layers = 1
    num_linear_layers = 1
    linear_out_channels = [5, 5] 
    device = "cuda"
    num_epochs = 10
    early_stop = True
    patience = 5