import logging
import os
import sys
import vtk
import slicer
from DICOMLib import DICOMUtils

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
        # Path to the DICOM directory
        dicomDirectory = r'C:\Users\pratt\OneDrive\Documents\Sample Images for Sequence Registration\manifest-1720549499368\CPTAC-CM\C3L-00629\04-23-2000-NA-MR BRAIN WOW CONTRAST-18837'
        fixedVolumes = []

        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(dicomDirectory, db)
            patientUIDs = db.patients()
            for patientUID in patientUIDs:
                fixedVolumes.extend(DICOMUtils.loadPatientByUID(patientUID))
        
        movingVolume = slicer.util.getNode(pattern='A1_grayT2')
        
        # Variables
        images = len(fixedVolumes)

        regType = input("What type of registration do you want to use? Type 'R' for Rigid, 'B' for BSpline, and 'A' for Affine. ")
        if regType == "R":
            for x in range(images):
                # Create output image and transformation for resampling
                resampledFixedVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
                doBrainResampling(fixedVolumes[x], movingVolume, resampledFixedVolume)
                
                # Create output image and transformation for registration
                outTrans = slicer.mrmlScene.CreateNodeByClass("vtkMRMLTransformNode")
                slicer.mrmlScene.AddNode(outTrans)
                outImg = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")
                slicer.mrmlScene.AddNode(outImg)
                doRigidRegistration(resampledFixedVolume, movingVolume, outImg, outTrans)
        elif regType == "A":
            for x in range(images):
                # Create output image and transformation for resampling
                resampledFixedVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
                doBrainResampling(fixedVolumes[x], movingVolume, resampledFixedVolume)
                
                # Create output image and transformation for registration
                outTrans = slicer.mrmlScene.CreateNodeByClass("vtkMRMLTransformNode")
                slicer.mrmlScene.AddNode(outTrans)
                outImg = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")
                slicer.mrmlScene.AddNode(outImg)
                doAffineRegistration(resampledFixedVolume, movingVolume, outImg, outTrans)
        elif regType == "B":
            for x in range(images):
                # Create output image and transformation for resampling
                resampledFixedVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
                doBrainResampling(fixedVolumes[x], movingVolume, resampledFixedVolume)
                
                # Create output image and transformation for registration
                bsplineTransform = slicer.mrmlScene.CreateNodeByClass("vtkMRMLTransformNode")
                slicer.mrmlScene.AddNode(bsplineTransform)
                finalImage = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")
                slicer.mrmlScene.AddNode(finalImage)
                doBSplineRegistration(resampledFixedVolume, movingVolume, finalImage, bsplineTransform)
        else:
            print("you dun goofed")
    else:
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
