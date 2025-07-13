#!/usr/bin/env python3
"""
Inference script for protein-ligand binding affinity prediction
Single prediction: python inference.py --protein protein.pdb --ligand ligand.sdf --model model.pt
Batch prediction: python inference.py --input_file pairs.txt --model model.pt --output results.csv
"""

import torch
import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from rdkit.Chem.rdmolfiles import MolFromPDBFile
from rdkit.Chem import SDMolSupplier
from torch_geometric.data import Data, Batch

# Import project modules
from model import Binding_Affinity_Predictor
from utils import ligand_to_graph, protein_to_graph, covalent_and_intermolecular_interactions_graph
from config import Config

def load_model(model_path, config):
    """Load the trained model"""
    print(f"Loading model from {model_path}")
    
    # Load the saved model
    loaded_model = torch.load(model_path, map_location=config.device, weights_only=False)
    
    # Check if it's a state dict or a complete model
    if isinstance(loaded_model, dict):
        # It's a state dict, create model and load state dict
        model = Binding_Affinity_Predictor(
            config.in_channels, 
            config.num_gnn_layers, 
            config.num_linear_layers, 
            config.linear_out_channels
        )
        model.load_state_dict(loaded_model)
    else:
        # It's a complete model object
        model = loaded_model
    
    model.to(config.device)
    model.eval()
    
    print("Model loaded successfully!")
    return model

def process_protein_ligand_pair(protein_path, ligand_path):
    """Process protein PDB and ligand SDF files into graph representation"""
    try:
        # Load protein
        protein = MolFromPDBFile(protein_path, sanitize=False)
        if protein is None:
            raise ValueError(f"Failed to load protein from {protein_path}")
        
        # Load ligand
        ligand = next(SDMolSupplier(ligand_path, sanitize=False))
        if ligand is None:
            raise ValueError(f"Failed to load ligand from {ligand_path}")
        
        # Convert to graphs
        ligand_graph = ligand_to_graph(ligand)
        protein_graph = protein_to_graph(protein)
        
        # Combine into protein-ligand complex graph
        complex_graph = covalent_and_intermolecular_interactions_graph(ligand_graph, protein_graph)
        
        # Create PyTorch Geometric Data object
        data = Data()
        data.__num_nodes__ = complex_graph["num_nodes"]
        data.x = torch.from_numpy(complex_graph["node_feat"]).to(torch.float32)
        
        edge_index = torch.from_numpy(complex_graph["edge_index"]).to(torch.long)
        data["edge_index_1"] = edge_index[:, :complex_graph["num_covalent_bonds"]]
        data["edge_index_2"] = edge_index
        data["edge_weight"] = torch.from_numpy(complex_graph["edge_weight"]).to(torch.float32)
        
        return data, None
    
    except Exception as e:
        return None, str(e)

def predict_binding_affinity(model, data, device):
    """Predict binding affinity for a protein-ligand complex"""
    try:
        # Create batch (single sample)
        batch = Batch.from_data_list([data]).to(device)
        
        # Predict
        with torch.no_grad():
            prediction = model(batch)
        
        # Convert to float
        affinity = prediction.cpu().numpy()[0][0]
        return affinity, None
    
    except Exception as e:
        return None, str(e)

def load_pairs_from_file(input_file):
    """Load protein-ligand pairs from input file"""
    pairs = []
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    protein_path = parts[0]
                    ligand_path = parts[1]
                    pair_id = parts[2] if len(parts) > 2 else f"pair_{line_num}"
                    pairs.append((protein_path, ligand_path, pair_id))
                else:
                    print(f"Warning: Invalid format at line {line_num}: {line}")
    
    return pairs

def single_prediction(args, model, config):
    """Handle single protein-ligand prediction"""
    print(f"Single prediction mode")
    print(f"Processing protein: {args.protein}")
    print(f"Processing ligand: {args.ligand}")
    
    # Process protein-ligand pair
    data, error = process_protein_ligand_pair(args.protein, args.ligand)
    if error:
        print(f"Error processing files: {error}")
        return 1
    
    # Print graph information
    print(f"Graph created successfully:")
    print(f"  - Number of nodes: {data.__num_nodes__}")
    print(f"  - Number of edges: {data['edge_index_2'].shape[1]}")
    print(f"  - Number of covalent bonds: {data['edge_index_1'].shape[1]}")
    print(f"  - Number of intermolecular interactions: {data['edge_index_2'].shape[1] - data['edge_index_1'].shape[1]}")
    
    # Predict binding affinity
    print("Predicting binding affinity...")
    affinity, error = predict_binding_affinity(model, data, config.device)
    if error:
        print(f"Error during prediction: {error}")
        return 1
    
    # Print results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Protein: {args.protein}")
    print(f"Ligand: {args.ligand}")
    print(f"Predicted Binding Affinity: {affinity:.4f}")
    print("="*50)
    
    # Additional interpretation
    print("\nInterpretation:")
    print(f"The predicted binding affinity value is {affinity:.4f}")
    print("Note: This value represents the log(Ki) or log(Kd) of the binding interaction.")
    print("Lower values indicate stronger binding affinity.")
    
    return 0

def batch_prediction(args, model, config):
    """Handle batch protein-ligand prediction"""
    print(f"Batch prediction mode")
    
    # Load protein-ligand pairs
    print(f"Loading pairs from {args.input_file}")
    pairs = load_pairs_from_file(args.input_file)
    print(f"Found {len(pairs)} protein-ligand pairs")
    
    # Process each pair
    results = []
    successful_predictions = 0
    
    for protein_path, ligand_path, pair_id in tqdm(pairs, desc="Processing pairs"):
        result = {
            'pair_id': pair_id,
            'protein_path': protein_path,
            'ligand_path': ligand_path,
            'predicted_affinity': None,
            'status': 'success',
            'error_message': None
        }
        
        # Check if files exist
        if not os.path.exists(protein_path):
            result['status'] = 'error'
            result['error_message'] = f"Protein file not found: {protein_path}"
            results.append(result)
            continue
        
        if not os.path.exists(ligand_path):
            result['status'] = 'error'
            result['error_message'] = f"Ligand file not found: {ligand_path}"
            results.append(result)
            continue
        
        # Process protein-ligand pair
        data, error = process_protein_ligand_pair(protein_path, ligand_path)
        if error:
            result['status'] = 'error'
            result['error_message'] = f"Processing error: {error}"
            results.append(result)
            continue
        
        # Predict binding affinity
        affinity, error = predict_binding_affinity(model, data, config.device)
        if error:
            result['status'] = 'error'
            result['error_message'] = f"Prediction error: {error}"
            results.append(result)
            continue
        
        result['predicted_affinity'] = affinity
        successful_predictions += 1
        results.append(result)
    
    # Save results
    output_file = args.output if args.output else 'batch_results.csv'
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n" + "="*50)
    print("BATCH PREDICTION RESULTS")
    print("="*50)
    print(f"Total pairs processed: {len(pairs)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {len(pairs) - successful_predictions}")
    print(f"Results saved to: {output_file}")
    print("="*50)
    
    # Show some statistics
    successful_results = df[df['status'] == 'success']
    if len(successful_results) > 0:
        print(f"\nAffinity Statistics:")
        print(f"Mean: {successful_results['predicted_affinity'].mean():.4f}")
        print(f"Std:  {successful_results['predicted_affinity'].std():.4f}")
        print(f"Min:  {successful_results['predicted_affinity'].min():.4f}")
        print(f"Max:  {successful_results['predicted_affinity'].max():.4f}")
    
    # Show failed cases
    failed_results = df[df['status'] == 'error']
    if len(failed_results) > 0:
        print(f"\nFailed Cases:")
        for idx, row in failed_results.iterrows():
            print(f"  {row['pair_id']}: {row['error_message']}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Predict protein-ligand binding affinity')
    
    # Single prediction mode
    parser.add_argument('--protein', help='Path to protein PDB file (for single prediction)')
    parser.add_argument('--ligand', help='Path to ligand SDF file (for single prediction)')
    
    # Batch prediction mode
    parser.add_argument('--input_file' ,help='Input file with protein-ligand pairs (tab-separated, for batch prediction)')
    parser.add_argument('--output', help='Output CSV file for batch results (default: batch_results.csv)')
    
    # Common arguments
    parser.add_argument('--model', default='./checkpoint/best_model.pt', help='Path to trained model (default: ./checkpoint/best_model.pt)')
    parser.add_argument('--config', default=None, help='Path to config file (optional)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_file:
        # Batch mode
        if args.protein or args.ligand:
            print("Warning: --protein and --ligand arguments are ignored in batch mode")
        mode = 'batch'
    elif args.protein and args.ligand:
        # Single mode
        mode = 'single'
    else:
        print("Error: Either specify --protein and --ligand for single prediction, or --input_file for batch prediction")
        parser.print_help()
        return 1
    
    # Load configuration
    config = Config()
    
    try:
        # Load model
        model = load_model(args.model, config)
        
        # Run prediction based on mode
        if mode == 'single':
            return single_prediction(args, model, config)
        else:
            return batch_prediction(args, model, config)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 