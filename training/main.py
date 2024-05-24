import os


os.system("nnUNetv2_plan_and_preprocess -d %s --verify_dataset_integrity -pl nnUNetPlannerResEncL" % (704))
os.system("nUNetv2_train %s  3d_fullres %s -p nnUNetResEncUNetLPlans" % (704, 0))
