import logging
import os
import sys
import vtk
import slicer
import numpy as np
import SimpleITK as sitk
import sitkUtils
from statistics import mean
from scipy.spatial.distance import dice
from DICOMLib import DICOMUtils


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
    print(f"PCC: {pcc}")

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
    
    print(f"Dice Similarity Coefficient: {dsc}")

def main():
    movingVolumes = []

    regType = input("What type of registration do you want to use? Type 'R' for Rigid, 'B' for BSpline, and 'A' for Affine. ")
    if regType not in ("R", "A", "B"):
        print("Invalid registration type.")
        return
    
    batchReg = input("If you are doing batch registration, press 'y'.")
    if batchReg == "y":
        dicomDirectory = r'C:\Users\pratt\Downloads\raw'

        fixedVolume = slicer.util.getNode(pattern='A1_grayT1')
        
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
            rescaleVolumeIntensities(fixedVolume)
            doPCC(outImg, fixedVolume)
            # doMI(fixedVolume, movingVolume, outTrans)
            doDSC(fixedVolume, outImg)
            print(f"Registration output name: {movingVolume.GetName()}_outputVolume")
            print("▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩▩")
            print()

    elif batchReg == "n":
        fixedVolume = slicer.util.getNode(pattern='A1_grayT1')
        movingVolume = slicer.util.getNode(pattern='4: 3D_AX_T1_precontrast')
        
        # Resample fixed volume
        resampledMovingVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        doBrainResampling(movingVolume, fixedVolume, resampledMovingVolume)

        outTrans = slicer.mrmlScene.CreateNodeByClass("vtkMRMLTransformNode")
        slicer.mrmlScene.AddNode(outTrans)
        outImg = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")
        slicer.mrmlScene.AddNode(outImg)

        if regType == "R":
            doRigidRegistration(fixedVolume, resampledMovingVolume, outImg, outTrans)
        elif regType == "A":
            doAffineRegistration(fixedVolume, resampledMovingVolume, outImg, outTrans)
        elif regType == "B":
            doBSplineRegistration(fixedVolume, resampledMovingVolume, outImg, outTrans)
        else:
            print("Invalid registration type")
        
        rescaleVolumeIntensities(fixedVolume)
        doPCC(outImg, fixedVolume)
        # doMI(fixedVolume, movingVolume, outTrans)
        doDSC(fixedVolume, outImg)
        
# Execute the main function with command-line arguments
if __name__ == "__main__":
    main()
