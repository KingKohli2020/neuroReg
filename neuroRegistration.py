import logging
import os
import sys
import vtk
import slicer
from DICOMLib import DICOMUtils

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

def rescaleVolumeIntensities(volumeNode):
    if isinstance(volumeNode, str):
        volumeNode = slicer.util.getNode(volumeNode)
    
    if not volumeNode or not volumeNode.GetImageData():
        raise ValueError("Invalid volume node provided.")
    
    import SimpleITK as sitk
    import sitkUtils
    
    sitkVolume = sitkUtils.PullVolumeFromSlicer(volumeNode)
    rescaler = sitk.RescaleIntensityImageFilter()
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)
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
    print("Done BSpline Registration")

def doAffineRegistration(fixedVolume, movingVolume, finalImage, outputTransform):
    paramsAffine = {'fixedVolume': fixedVolume,
                    'movingVolume': movingVolume,
                    'outputTransform': outputTransform,
                    'outputVolume': finalImage,
                    'maskProcessingMode': "NOMASK",
                    'useAffine': True}
    slicer.cli.run(slicer.modules.brainsfit, None, paramsAffine, wait_for_completion=True)
    print("Done Affine Registration")

def doBrainResampling(inputVolume, referenceVolume, outputVolume, interpolationMode='Linear'):
    paramsBrainResample = {
        'inputVolume': inputVolume,
        'referenceVolume': referenceVolume,
        'outputVolume': outputVolume,
        'interpolationMode': interpolationMode
    }
    cliNode = slicer.cli.run(slicer.modules.brainsresample, None, paramsBrainResample, wait_for_completion=True)
    if cliNode.GetStatusString() != 'Completed':
        logging.error("Resample Image (BRAINS) failed: " + cliNode.GetStatusString())
    else:
        logging.info("Resample Image (BRAINS) completed successfully.")

def main():
    batchReg = input("If you are doing batch registration, press 'y'.")
    fixedVolumes = []
    if batchReg == "y":
        dicomDirectory = r'C:\Users\pratt\Downloads\raw'

        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(dicomDirectory, db)
            patientUIDs = db.patients()
            for patientUID in patientUIDs:
                fixedVolumes.extend(DICOMUtils.loadPatientByUID(patientUID))
        
        movingVolume = slicer.util.getNode(pattern='A1_grayT1')
        images = len(fixedVolumes)

        regType = input("What type of registration do you want to use? Type 'R' for Rigid, 'B' for BSpline, and 'A' for Affine. ")
        if regType not in ("R", "A", "B"):
            print("Invalid registration type.")
            return

        for x in range(images):
            fixedVolume = fixedVolumes[x]

            # Ensure fixedVolume is a volume node
            if isinstance(fixedVolume, str):
                fixedVolume = slicer.util.getNode(fixedVolume)

            if not isinstance(fixedVolume, slicer.vtkMRMLScalarVolumeNode):
                print(f"Invalid volume node at index {x}. It is not a vtkMRMLScalarVolumeNode. Skipping this volume.")
                continue

            fixedVolume = convertVolumeScalarType(fixedVolume)
            # rescaleVolumeIntensities(fixedVolume)
            # alignVolumeOrigin(fixedVolume, movingVolume)
            
            doBrainResampling(fixedVolume, movingVolume, fixedVolume)
            
            outImg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{fixedVolume.GetName()}_outputVolume")
            outTrans = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", f"{fixedVolume.GetName()}_outputTransform")

            if regType == "R":
                doRigidRegistration(fixedVolume, movingVolume, outImg, outTrans)
            elif regType == "A":
                doAffineRegistration(fixedVolume, movingVolume, outImg, outTrans)
            elif regType == "B":
                doBSplineRegistration(fixedVolume, movingVolume, outImg, outTrans)

    
    elif regType == "n":
        fixedVolume = slicer.util.getNode(pattern='701: sT1W_3D_TFE_AX PRE')
        movingVolume = slicer.util.getNode(pattern='A1_grayT2')
        
        # Resample fixed volume
        resampledFixedVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        doBrainResampling(fixedVolume, movingVolume, resampledFixedVolume)

        regType = input("What type of registration do you want to use? Type 'R' for Rigid, 'B' for BSpline, and 'A' for Affine. ")
        if regType == "R":
            # Create output image and transformation
            outTrans = slicer.mrmlScene.CreateNodeByClass("vtkMRMLTransformNode")
            slicer.mrmlScene.AddNode(outTrans)
            outImg = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")
            slicer.mrmlScene.AddNode(outImg)
            doRigidRegistration(resampledFixedVolume, movingVolume, outImg, outTrans)
        elif regType == "A":
            # Create output image and transformation
            outTrans = slicer.mrmlScene.CreateNodeByClass("vtkMRMLTransformNode")
            slicer.mrmlScene.AddNode(outTrans)
            outImg = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")
            slicer.mrmlScene.AddNode(outImg)
            doAffineRegistration(resampledFixedVolume, movingVolume, outImg, outTrans)
        elif regType == "B":
            # Create output image and transformation
            bsplineTransform = slicer.mrmlScene.CreateNodeByClass("vtkMRMLTransformNode")
            slicer.mrmlScene.AddNode(bsplineTransform)
            finalImage = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")
            slicer.mrmlScene.AddNode(finalImage)
            doBSplineRegistration(resampledFixedVolume, movingVolume, finalImage, bsplineTransform)
        else:
            print("you dun goofed")

# Execute the main function with command-line arguments
if __name__ == "__main__":
    main()
