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

def main():
    batchReg = input("If you are doing batch registration, press 'y'.")
    fixedVolume = []
    if batchReg == "y":
        # Path to the DICOM directory
        dicomDirectory = r'C:\Users\pratt\OneDrive\Documents\Sample Images for Sequence Registration\manifest-1720549499368\CPTAC-CM\C3L-00629\04-23-2000-NA-MR BRAIN WOW CONTRAST-18837'
        fixedVolume = []

        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(dicomDirectory, db)
            patientUIDs = db.patients()
            for patientUID in patientUIDs:
                fixedVolume.extend(DICOMUtils.loadPatientByUID(patientUID))
        
        #variables
        images = len(fixedVolume)

        
    else:
        fixedVolume = slicer.util.getNode(pattern='701: sT1W_3D_TFE_AX PRE')
        #fixedVolume = slicer.util.getNode(pattern='3: MR ep2d_perf 12 CC BOLUS - 45 frames Volume Sequence by AcquisitionTime [0]')

        images = 1
    
    movingVolume = slicer.util.getNode(pattern='A1_grayT2')


    regType = input("What type of registration do you want to use? Type 'R' for Rigid, 'B' for BSpline, and 'A' for Affine. ")
    if regType == "R":
        for x in range(images):
            #create output image and transformation
            outTrans = slicer.mrmlScene.CreateNodeByClass("vtkMRMLTransformNode")
            slicer.mrmlScene.AddNode(outTrans)
            outImg = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")
            slicer.mrmlScene.AddNode(outImg)
            doRigidRegistration(fixedVolume, movingVolume, outImg, outTrans)
    elif regType == "A":
        for x in range(images):
            #create output image and transformation
            outTrans = slicer.mrmlScene.CreateNodeByClass("vtkMRMLTransformNode")
            slicer.mrmlScene.AddNode(outTrans)
            outImg = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")
            slicer.mrmlScene.AddNode(outImg)
            doAffineRegistration(fixedVolume, movingVolume, outImg, outTrans)
    elif regType == "B":
        for x in range(images):
            #create output image and transformation
            bsplineTransform = slicer.mrmlScene.CreateNodeByClass("vtkMRMLTransformNode")
            slicer.mrmlScene.AddNode(bsplineTransform)
            finalImage = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")
            slicer.mrmlScene.AddNode(finalImage)
            doBSplineRegistration(fixedVolume, movingVolume, finalImage, bsplineTransform)
    else:
          print("you dun goofed")
# Execute the main function with command-line arguments
if __name__ == "__main__":
        main()
