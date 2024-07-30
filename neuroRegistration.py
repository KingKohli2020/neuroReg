import logging
import os
import sys
import vtk
import slicer
import joblib
import pydicom
import numpy as np
import pandas as pd
import SimpleITK as sitk
import sitkUtils
from statistics import mean
from scipy.spatial.distance import dice
from DICOMLib import DICOMUtils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict


# data = {
#     "Quality": ["Good", "Bad", "Bad", "Bad", "Good"],
#     "Mean": [316.7, 462.5, 293.3, 430.5, 933.2],
#     "Std Dev": [56.4, 195.2, 30.4, 130.5, 370.1],
#     "PCC": [0.01916, 0.01095, -0.02534, -0.00042, 0.00070],
#     "DSC": [0.76985, 0.32232, 0.54778, 0.16228, 0.36639]
# }

# df = pd.DataFrame(data)

# # Encode the 'Quality' column
# label_encoder = LabelEncoder()
# df["Quality"] = label_encoder.fit_transform(df["Quality"])

# # Split the data into features and labels
# X = df.drop("Quality", axis=1)
# y = df["Quality"]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")


# joblib.dump(model, "quality_model.pkl")
# joblib.dump(label_encoder, "label_encoder.pkl")

model = joblib.load("quality_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def classify_registration_with_model(mean, std_dev, pcc, dsc, model, label_encoder):
    new_data = pd.DataFrame({
        "Mean": [mean],
        "Std Dev": [std_dev],
        "PCC": [pcc],
        "DSC": [dsc]
    })

    prediction = model.predict(new_data)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

def rescaleVolumeIntensities(volumeNode):
    if isinstance(volumeNode, str):
        volumeNode = slicer.util.getNode(volumeNode)
    
    if not volumeNode or not volumeNode.GetImageData():
        raise ValueError("Invalid volume node provided.")
    
    
    sitkVolume = sitkUtils.PullVolumeFromSlicer(volumeNode)
    rescaler = sitk.RescaleIntensityImageFilter()
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(1227.34)
    normalizedSitkVolume = rescaler.Execute(sitkVolume)
    sitkUtils.PushVolumeToSlicer(normalizedSitkVolume, volumeNode)
    return volumeNode

def alignVolumeOrigin(volumeNode, referenceVolumeNode):
    volumeNode.SetOrigin(referenceVolumeNode.GetOrigin())

def doRigidRegistration(fixedVolume, movingVolume, finalImage, outputTransform):
    paramsRigid = {'fixedVolume': fixedVolume,
                'movingVolume': movingVolume,
                'outputTransform': outputTransform,
                'outputVolume': finalImage,
                'maskProcessingMode': "NOMASK",
                'initializeTransformMode':'useCenterOfHeadAlign',
                'useRigid': True}
    slicer.cli.run(slicer.modules.brainsfit, None, paramsRigid, wait_for_completion=True)
    print()
    print("Done Rigid Registration")

def doBSplineRegistration(fixedVolume, movingVolume, finalImage, bsplineTransform):
    paramsBSpline = {'fixedVolume': fixedVolume,
                    'movingVolume': movingVolume,
                    'outputVolume': finalImage,
                    'bsplineTransform': bsplineTransform,
                    'useROIBSpline': False,
                    'useBSpline': True,
                    'splineGridSize': "14, 10, 12",
                    'maskProcessing': "NOMASK",
                    'minimumStepLength': "0.005",
                    'maximumStepLength': "0.1",}

    slicer.cli.run(slicer.modules.brainsfit, None, paramsBSpline, wait_for_completion=True)
    print()
    print("Done BSpline Registration")

def doAffineRegistration(fixedVolume, movingVolume, finalImage, outputTransform):
    paramsAffine = {'fixedVolume': fixedVolume,
                    'movingVolume': movingVolume,
                    'outputTransform': outputTransform,
                    'outputVolume': finalImage,
                    'maskProcessingMode': "NOMASK",
                    'useAffine': True}
    slicer.cli.run(slicer.modules.brainsfit, None, paramsAffine, wait_for_completion=True)
    print()
    print("Done Affine Registration")

def convertVolumeScalarType(volumeNode, targetType='Float'):
    params = {
        'InputVolume': volumeNode,
        'OutputVolume': slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode'),
        'Type': targetType
    }
    cliNode = slicer.cli.runSync(slicer.modules.castscalarvolume, None, params)
    if cliNode.GetStatusString() != 'Completed':
        logging.error("Cast Scalar Volume failed: " + cliNode.GetStatusString())
        raise RuntimeError(f"Scalar volume casting failed with status: {cliNode.GetStatusString()}")
    return params['OutputVolume']

def doBrainResampling(inputVolume, referenceVolume, outputVolume, interpolationMode='Linear'):
    paramsBrainResample = {
        'inputVolume': inputVolume,
        'referenceVolume': referenceVolume,
        'outputVolume': outputVolume,
        'interpolationMode': interpolationMode
    }
    cliNode = slicer.cli.runSync(slicer.modules.brainsresample, None, paramsBrainResample)
    if cliNode.GetStatusString() != 'Completed':
        logging.error("Resample Image (BRAINS) failed: " + cliNode.GetStatusString())
        raise RuntimeError(f"Resample Image (BRAINS) failed with status: {cliNode.GetStatusString()}")
    else:
        logging.info("Resample Image (BRAINS) completed successfully.")

def resampleVolumeToReference(volumeNode, referenceVolumeNode):
    resampledVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    params = {
        'inputVolume': volumeNode,
        'referenceVolume': referenceVolumeNode,
        'outputVolume': resampledVolumeNode,
        'interpolationMode': 'Linear'
    }
    cliNode = slicer.cli.runSync(slicer.modules.resamplescalarvolume, None, params)
    if cliNode.GetStatusString() != 'Completed':
        logging.error("Resample Volume failed: " + cliNode.GetStatusString())
        raise RuntimeError(f"Volume resampling failed with status: {cliNode.GetStatusString()}")
    return resampledVolumeNode

def doPCC(outImg, fixedVolume):
    # Resample outImg to match fixedVolume's dimensions
    # resampledOutImg = resampleVolumeToReference(outImg, fixedVolume)

    registeredVoxelArray = slicer.util.arrayFromVolume(outImg)
    atlasVoxelArray = slicer.util.arrayFromVolume(fixedVolume)

    # Ensure arrays have the same shape
    # if registeredVoxelArray.shape != atlasVoxelArray.shape:
    #     raise ValueError("Registered and fixed volumes must have the same dimensions.")

    # Thresholding to focus on specific regions
    threshold = 260
    regIntensities = registeredVoxelArray[registeredVoxelArray > threshold]
    atlIntensities = atlasVoxelArray[atlasVoxelArray > threshold]

    # Ensure the same number of voxels above threshold
    if len(regIntensities) != len(atlIntensities):
        min_length = min(len(regIntensities), len(atlIntensities))
        regIntensities = regIntensities[:min_length]
        atlIntensities = atlIntensities[:min_length]

    # Calculate Pearson Correlation Coefficient
    regMean = np.mean(regIntensities)
    regStdDev = np.std(regIntensities)
    atlMean = np.mean(atlIntensities)
    atlStdDev = np.std(atlIntensities)

    if regStdDev == 0 or atlStdDev == 0:
        raise ValueError("Standard deviation of intensities must not be zero.")

    normalizedRegIntensities = (regIntensities - regMean) / regStdDev
    normalizedAtlIntensities = (atlIntensities - atlMean) / atlStdDev

    pcc = np.mean(normalizedRegIntensities * normalizedAtlIntensities)
    # print(f"PCC: {pcc}")
    # print(f"Mean Intensity: {regMean}")
    # print(f"Intensity Standard Deviation: {regStdDev}")
    return pcc, regMean, regStdDev

def doMI(fixedVolume, movingVolume, outTrans):
    metric = sitk.ImageRegistrationMethod()
    metric.SetMetricAsMattesMutualInformation()
    mi_value = metric.MetricEvaluate(fixedVolume, movingVolume, outTrans)
    print(f"Mutual Information: {mi_value}")        

def doDSC(fixedVolume, movingVolume):
    # Convert vtkMRMLScalarVolumeNode to SimpleITK Image
    fixedImage = sitkUtils.PullVolumeFromSlicer(fixedVolume)
    movingImage = sitkUtils.PullVolumeFromSlicer(movingVolume)
    
    # Ensure the images are binary
    fixedBinary = sitk.BinaryThreshold(fixedImage, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
    movingBinary = sitk.BinaryThreshold(movingImage, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
    
    # Compute Dice Similarity Coefficient
    diceFilter = sitk.LabelOverlapMeasuresImageFilter()
    diceFilter.Execute(fixedBinary, movingBinary)
    dsc = diceFilter.GetDiceCoefficient()
    
    # print(f"Dice Similarity Coefficient: {dsc}")
    return dsc

def extract_dicom_metadata(directory_path):
    metadata = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.dcm'):
                file_path = os.path.join(root, file)
                ds = pydicom.dcmread(file_path)
                metadata.append({
                    'file_path': file_path,
                    'StudyDate': getattr(ds, 'StudyDate', ''),
                    'Modality': getattr(ds, 'Modality', ''),
                    'SeriesDescription': getattr(ds, 'SeriesDescription', ''),
                    'InstanceNumber': getattr(ds, 'InstanceNumber', 0)
                })
    return metadata

def group_dicom_files_by_metadata(metadata):
    groups = defaultdict(list)
    for item in metadata:
        key = (item['StudyDate'], item['Modality'], item['SeriesDescription'])
        groups[key].append(item['file_path'])
    
    return groups

def load_grouped_dicoms(groups):
    loaded_volumes = []
    for key, file_paths in groups.items():
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(file_paths, db)
            loaded_nodes = []
            for patientUID in db.patients():
                loaded_nodes.extend(DICOMUtils.loadPatientByUID(patientUID))
            if loaded_nodes:
                loaded_volumes.extend(loaded_nodes)
                for node in loaded_nodes:
                    print(f"Loaded volume: {node.GetName()}")
    return loaded_volumes

def main():
    fixedVolume = slicer.util.getNode(pattern='A1_grayT1')
    movingVolumes = []
    # goodRegistrations = []

    regType = input("What type of registration do you want to use? Type 'R' for Rigid, 'B' for BSpline, and 'A' for Affine. ")
    if regType not in ("R", "A", "B"):
        print("Invalid registration type.")
        return
    
    batchReg = input("If you are doing batch registration, press 'y'.")
    if batchReg == "y":
        dicomDirectory = r"C:\Users\pratt\Downloads\raw"
        
        # metadata = extract_dicom_metadata(dicomDirectory)
        # groups = group_dicom_files_by_metadata(metadata)
        # movingVolumes = load_grouped_dicoms(groups)

        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(dicomDirectory, db)
            patientUIDs = db.patients()
            for patientUID in patientUIDs:
                movingVolumes.extend(DICOMUtils.loadPatientByUID(patientUID))
                
        for x in range(len(movingVolumes)):
            movingVolume = movingVolumes[x]

            # Ensure movingVolume is a volume node
            if isinstance(movingVolume, str):
                try:
                    movingVolume = slicer.util.getNode(movingVolume)
                except:
                    continue

            if not isinstance(movingVolume, slicer.vtkMRMLScalarVolumeNode):
                print(f"Invalid volume node at index {x}. It is not a vtkMRMLScalarVolumeNode. Skipping this volume.")
                continue

            # Adjust volume scalar type
            adjmovingVolume = convertVolumeScalarType(movingVolume)

            # recenter attempt
            # alignVolumeOrigin(movingVolume, fixedVolume)
            
            # Resample the adjusted volume
            # doBrainResampling(adjmovingVolume, fixedVolume, adjmovingVolume)
            
            # Add output nodes to the scene
            outImg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{movingVolume.GetName()}_outputVolume")
            outTrans = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", f"{movingVolume.GetName()}_outputTransform")

            # Perform registration based on selected type
            if regType == "R":
                doRigidRegistration(fixedVolume, adjmovingVolume, outImg, outTrans)
            elif regType == "A":
                doAffineRegistration(fixedVolume, adjmovingVolume, outImg, outTrans)
            elif regType == "B":
                doBSplineRegistration(fixedVolume, adjmovingVolume, outImg, outTrans)
            
            # conduct registration similarity measures
            pcc, avg, std = doPCC(outImg, fixedVolume)[0], doPCC(outImg, fixedVolume)[1], doPCC(outImg, fixedVolume)[2]
            # doMI(fixedVolume, movingVolume, outTrans)
            dsc = doDSC(fixedVolume, outImg)

            quality = classify_registration_with_model(avg, std, pcc, dsc, model, label_encoder)
            print(f"Quality: {quality.upper()}")
            print(f"Registration output name: {movingVolume.GetName()}_outputVolume")
            print("▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩")
            print()

            # if quality.upper() == "GOOD":
            #     goodRegistrations.append(outImg)

    elif batchReg == "n":
        movingVolume = slicer.util.getNode(pattern='image.0001.dcm')
        
        # Resample fixed volume
        resampledMovingVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        doBrainResampling(movingVolume, fixedVolume, resampledMovingVolume)

        outImg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{movingVolume.GetName()}_outputVolume")
        outTrans = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", f"{movingVolume.GetName()}_outputTransform")

        if regType == "R":
            doRigidRegistration(fixedVolume, resampledMovingVolume, outImg, outTrans)
        elif regType == "A":
            doAffineRegistration(fixedVolume, resampledMovingVolume, outImg, outTrans)
        elif regType == "B":
            doBSplineRegistration(fixedVolume, resampledMovingVolume, outImg, outTrans)
        else:
            print("Invalid registration type")
        
        rescaleVolumeIntensities(fixedVolume)
        pcc, avg, std = doPCC(outImg, fixedVolume)[0], doPCC(outImg, fixedVolume)[1], doPCC(outImg, fixedVolume)[2]
        # doMI(fixedVolume, movingVolume, outTrans)
        dsc = doDSC(fixedVolume, outImg)

        quality = classify_registration_with_model(avg, std, pcc, dsc, model, label_encoder)
        print(f"Quality: {quality.upper()}")

        # if quality.upper() == "GOOD":
        #         goodRegistrations.append(outImg)
    
    # Save "GOOD" registrations to a file
    # if goodRegistrations:
    #     output_directory = r"C:\Users\pratt\Downloads\goodRegistrations"
    #     if not os.path.exists(output_directory):
    #         os.makedirs(output_directory)
        
    #     for i, volumeNode in enumerate(goodRegistrations):
    #         output_path = os.path.join(output_directory, outImg.GetName())  # Specify the file extension
    #         success = slicer.util.saveNode(volumeNode, output_path)
    #         if success:
    #             print(f"Saved {volumeNode.GetName()} to {output_path}")
    #         else:
    #             print(f"Failed to save {volumeNode.GetName()} to {output_path}")

        
# Execute the main function with command-line arguments
if __name__ == "__main__":
    main()
