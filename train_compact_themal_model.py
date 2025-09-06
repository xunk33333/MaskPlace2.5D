#themal model from
#ATPlace2.5D: Analytical Thermal-Aware Chiplet Placement Framework for Large-Scale 2.5D-IC
import argparse
import os
from process_thermal import parse_intpsize_file,parse_power_file,parse_pl_file
from compact_themal_model import ChipletThermalModel
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from gf_layer4_tool import tri_panel, compute_metrics
import time
grid = 64

def train(case_name):
    TRAIN = 50

    ENV_TMP = 45  # Environment temperature in Celsius
    ENV_TMP_K = ENV_TMP + 273.15  # Convert to Kelvin
    case_dir = f"./cases/{case_name}"
    intpsize_path = os.path.join(case_dir, f"{case_name}.intpsize")
    intpsize, _ = parse_intpsize_file(intpsize_path)
    intpsize /= 1e3
    power_file = os.path.join(case_dir, f"{case_name}.power")
    power = parse_power_file(power_file)
    
    T_ground_truth = torch.zeros([TRAIN,1,grid,grid], dtype=torch.float32, device='cuda')
    dataset_x = np.zeros([TRAIN, len(power)], dtype=np.float32)  
    dataset_y = np.zeros([TRAIN, len(power)], dtype=np.float32)
    dataset_width = np.zeros([TRAIN, len(power)], dtype=np.float32)
    dataset_height = np.zeros([TRAIN, len(power)], dtype=np.float32)
    dataset_power = np.zeros([TRAIN, len(power)], dtype=np.float32)
    
    X, Y = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='xy')
    x_input = X.flatten().float().repeat(TRAIN, 1) * intpsize / grid
    y_input = Y.flatten().float().repeat(TRAIN, 1) * intpsize / grid
    for i in range(TRAIN):
        gt_path = f'dataset/{case_name}/gen_dataset_{i+1}'
        np_ground_truth = np.flipud(np.loadtxt(gt_path + ".grid.steady", usecols=[1]).reshape(grid,grid)).copy()
        T_ground_truth[i, 0] = torch.tensor(np_ground_truth, dtype=torch.float32, device='cuda').view(grid, grid)
        
        pl_file = os.path.join(case_dir, f"{case_name}_{i+1}.pl")
        chip_names, positions, widths, heights = parse_pl_file(pl_file)
        positions = np.array(positions).transpose()
        widths = np.array(widths)
        heights = np.array(heights)
        if i == 0:
            power = [power.get(name, 0.0) for name in chip_names]
        
        dataset_x[i] = np.round(positions[0]/1e3)
        dataset_y[i] = np.round(positions[1]/1e3)
        dataset_width[i] = np.array(widths/1e3)
        dataset_height[i] = np.array(heights/1e3)
        dataset_power[i] = np.array(power, dtype=np.float32)


    lr = 0.1
    steps = 10000

    model = ChipletThermalModel(len(power)).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)
    loss_fn = torch.nn.MSELoss()

    x_input = x_input.to('cuda')
    y_input = y_input.to('cuda')
    dataset_x = torch.tensor(dataset_x, dtype=torch.float32, device='cuda')
    dataset_y = torch.tensor(dataset_y, dtype=torch.float32, device='cuda')
    dataset_width = torch.tensor(dataset_width, dtype=torch.float32, device='cuda')
    dataset_height = torch.tensor(dataset_height, dtype=torch.float32, device='cuda') 
    dataset_power = torch.tensor(dataset_power, dtype=torch.float32, device ='cuda')
    for step in tqdm(range(steps)):
        optimizer.zero_grad()
        T_pred = model(x_input, y_input, dataset_x, dataset_y, dataset_width, dataset_height, dataset_power, grid) + ENV_TMP_K
        loss = loss_fn(T_pred, T_ground_truth.view(TRAIN, -1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"Step {step:4d}:, Loss = {loss.item():.6f}")
    torch.save(model.state_dict(), f'model/{case_name}_thermal_model.pth')
    print(f"\n✅ Training complete. Model saved to output/{case_name}/thermal_model.pth")

def test(case_name):
    ENV_TMP = 45  # Environment temperature in Celsius
    ENV_TMP_K = ENV_TMP + 273.15  # Convert to Kelvin
    case_dir = f"./cases/{case_name}"
    intpsize_path = os.path.join(case_dir, f"{case_name}.intpsize")
    intpsize, _ = parse_intpsize_file(intpsize_path)
    intpsize /= 1e3
    power_file = os.path.join(case_dir, f"{case_name}.power")
    power = parse_power_file(power_file)
    print(f'Case:{case_name}')

    model_path = f'model/{case_name}_thermal_model.pth'
    model = ChipletThermalModel(len(power))
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.to('cuda')
    X, Y = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='xy')
    x_input = X.flatten().float().view(1, -1).to('cuda') * intpsize / grid
    y_input = Y.flatten().float().view(1, -1).to('cuda') * intpsize / grid
    metrics_rows = []
    sum_time = 0.0
    with torch.no_grad():
        model.eval()
        for i in range(51, 201):
            gt_path = f'dataset/{case_name}/gen_dataset_{i}'
            pl_file = os.path.join(case_dir, f"{case_name}_{i}.pl")
            chip_names, positions, widths, heights = parse_pl_file(pl_file)
            positions = np.array(positions).transpose()
            widths = np.array(widths)
            heights = np.array(heights)
            if i == 51:
                power = [power.get(name, 0.0) for name in chip_names]
             
            dataset_x = torch.tensor(np.round(positions[0]/1e3), dtype=torch.float32, device='cuda').view(1, -1)
            dataset_y = torch.tensor(np.round(positions[1]/1e3) , dtype=torch.float32, device='cuda').view(1, -1)
            dataset_width = torch.tensor(np.array(widths/1e3), dtype=torch.float32, device='cuda').view(1, -1)
            dataset_height = torch.tensor(np.array(heights/1e3), dtype=torch.float32, device='cuda').view(1, -1)
            dataset_power = torch.tensor(np.array(power, dtype=np.float32)     , dtype=torch.float32, device='cuda').view(1, -1)

            start = time.time()
            T_pred = model(x_input, y_input, dataset_x, dataset_y, dataset_width, dataset_height, dataset_power, grid) + ENV_TMP_K
            end = time.time()
            print(f"Inference time for {i}: {end - start:.6f} seconds")
            sum_time += (end - start)
            T_pred_np = T_pred.cpu().numpy().reshape(grid, grid)
            T_ground_truth_np = np.flipud(np.loadtxt(gt_path + ".grid.steady", usecols=[1]).reshape(grid, grid)).copy()
            mae = np.mean(np.abs(T_pred_np - T_ground_truth_np))
            print(f"Test {i}: MAE = {mae:.4f} °C")
            tri_panel(T_ground_truth_np - 273.15,T_pred_np- 273.15, f"tmp/sid{i}_gt_pred_compact.png")
            
            m = compute_metrics(T_pred_np- 273.15, T_ground_truth_np- 273.15)
            metrics_rows.append((i, m["MAE"], m["RMSE"], m["MAPE"], m["CORR"],  m["PTE"]))
            print(f"Finshed processing gen_dataset_{i}.")
        print(f"Average inference time: {sum_time/150:.6f} seconds")       
    # Save metrics CSV
    import csv
    csv_path = os.path.join('tmp/', f"compact_metrics_auto_test_{case_name}.csv")
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["sid","MAE_C", "RMSE_C", "MAPE", "CORR", "PTE"])
        for row in metrics_rows:
            w.writerow(row)
        # Append mean row
        if metrics_rows:
            mean_mae = float(np.mean([r[1] for r in metrics_rows]))
            mean_rmse = float(np.mean([r[2] for r in metrics_rows]))
            mean_mape = float(np.mean([r[3] for r in metrics_rows]))
            mean_corr = float(np.nanmean([r[4] for r in metrics_rows]))
            mean_pte = float(np.mean([r[5] for r in metrics_rows]))
            w.writerow(["mean", mean_mae, mean_rmse, mean_mape, mean_corr, mean_pte])
if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--case', type=str, default="10", help='case name')
    arg.add_argument('--train',action='store_true', help='Set this flag to train the model')
    args = arg.parse_args()
    
    case_name = f"Case{args.case}"
    if args.train:
        start = time.time()
        train(case_name)
        end = time.time()
        print(f"Total training time: {end - start:.2f} seconds")
    test(case_name)  # Uncomment to run the test function