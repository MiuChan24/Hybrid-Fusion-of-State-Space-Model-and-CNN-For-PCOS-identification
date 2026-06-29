def evaluate_hybrid_ensemble(model_resnet, model_mamba, test_loader, device):
    all_labels = []
    all_probs = []
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # --- MUST BE INDENTED HERE (INSIDE THE FOR LOOP) ---
            resnet_out = model_resnet(images)
            mamba_out = model_mamba(images)
            ensemble_probs = torch.softmax((resnet_out + mamba_out) / 2.0, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(ensemble_probs.cpu().numpy())
            
    # --- MUST BE INDENTED HERE (OUTSIDE THE FOR LOOP) ---
    final_labels = np.array(all_labels)
    final_probs = np.concatenate(all_probs, axis=0)
    
    np.save('results_labels.npy', final_labels)
    np.save('results_probs.npy', final_probs)
    
    print("Successfully saved 61 labels and 61 probability predictions!")
    
    # ... (Keep whatever code you have below this for printing the classification report) ...
