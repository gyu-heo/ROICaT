
## Find network files
if names_networkFiles is None:
    names_networkFiles = {
        'params': 'params.json',
        'model': 'model.py',
        'state_dict': '.pth',
    }
paths_networkFiles = {}
paths_networkFiles['params'] = [p for p in paths_extracted if names_networkFiles['params'] in str(Path(p).name)][0]
# paths_networkFiles['model'] = [p for p in paths_extracted if names_networkFiles['model'] in str(Path(p).name)][0]
paths_networkFiles['model'] = '/home/josh/github-repos/ROICaT_paper/simclr_benchmark/C:\\\\Users\\\\Josh\\\\Downloads\\\\ROICaT\\\\cls/model.py'
paths_networkFiles['state_dict'] = [p for p in paths_extracted if names_networkFiles['state_dict'] in str(Path(p).name)][0]


sys.path.append(str(Path(paths_networkFiles['model']).parent.resolve()))
import model
print(f"Imported model from {paths_networkFiles['model']}") if _verbose else None

with open(paths_networkFiles['params']) as f:
    params_model = json.load(f)
    print(f"Loaded params_model from {paths_networkFiles['params']}") if _verbose else None
    net = model.make_model(fwd_version='head', **params_model)
    print(f"Generated network using params_model") if _verbose else None

net.load_state_dict(torch.load(Path(r'./C:\\Users\\Josh\\Downloads\\ROICaT\\cls')/('model_wo_pca_weights.pth'), map_location=torch.device(_device)))

net.pca_layer = torch.nn.Sequential(
    torch.nn.Linear(128, 128),
    torch.nn.Linear(128, 128)
)

net.pca_layer[0].weight = torch.nn.Parameter(torch.tensor(np.eye(128),dtype=torch.float32))
net.pca_layer[0].bias = torch.nn.Parameter(torch.tensor(-np_latent.mean(axis=0),dtype=torch.float32))
net.pca_layer[1].weight = torch.nn.Parameter(torch.tensor(pca.components_,dtype=torch.float32))
net.pca_layer[1].bias = torch.nn.Parameter(torch.tensor(np.zeros(128),dtype=torch.float32))

torch.save(net.state_dict(), Path(r'./C:\\Users\\Josh\\Downloads\\ROICaT\\cls')/('model_w_pca_weights.pth'))