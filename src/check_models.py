import timm
from pprint import pprint
from tqdm import tqdm

model_names = timm.list_models(pretrained=True)

# model_names = timm.list_models('*eff*')


pprint(model_names)
# for i, model in tqdm(enumerate(model_names)):
#     # model = 'cait_m36_384'
#     if i > 275:
#         m = timm.create_model(model, pretrained=True)
#     # pprint(m.default_cfg)


model = 'resnet152d'
m = timm.create_model(model, pretrained=True)
print(m)


pprint(m.default_cfg)