import logging
import os
import sys
import slicer
from typing import Annotated  # Import Annotated from typing module
from slicer.ScriptedLoadableModule import *
from slicer.parameterNodeWrapper import parameterNodeWrapper, WithinRange
from slicer import vtkMRMLScalarVolumeNode
import argparse

from SlicerDevelopmentToolboxUtils.mixins import ModuleLogicMixin, ModuleWidgetMixin
from SliceTrackerUtils.sessionData import *
from SliceTrackerUtils.constants import SliceTrackerConstants
from SlicerDevelopmentToolboxUtils.decorators import onReturnProcessEvents
#
# neuroRegistration
#

class neuroRegistration(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class"""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "neuroRegistration"
        self.parent.categories = ["Registration"]
        self.parent.dependencies = []
        self.parent.contributors = ["Prattay Bhattacharya (SNR Lab @ Harvard)"]
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#neuroRegistration">module documentation</a>.
"""
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        slicer.app.connect("startupCompleted()", registerSampleData)

def registerSampleData():
    """Add data sets to Sample Data module."""
    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

@parameterNodeWrapper
class neuroRegistrationParameterNode:
    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode

class neuroRegistrationLogic(ScriptedLoadableModuleLogic):
    
    counter = 1
    
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.registrationResult = None

    def _processParameterNode(self, parameterNode):
        if not self.registrationResult:
            self.registrationResult = RegistrationResult("01: RegistrationResult")
        result = self.registrationResult
        result.volumes.fixed = slicer.mrmlScene.GetNodeByID(parameterNode.GetAttribute('FixedImageNodeID'))
        result.labels.fixed = slicer.mrmlScene.GetNodeByID(parameterNode.GetAttribute('FixedLabelNodeID'))
        result.labels.moving = slicer.mrmlScene.GetNodeByID(parameterNode.GetAttribute('MovingLabelNodeID'))
        movingVolume = slicer.mrmlScene.GetNodeByID(parameterNode.GetAttribute('MovingImageNodeID'))
        result.volumes.moving = self.volumesLogic.CloneVolume(slicer.mrmlScene, movingVolume,
                                                            "temp-movingVolume_" + str(self.counter))
        self.counter += 1

        logging.debug("Fixed Image Name: %s" % result.volumes.fixed.GetName())
        logging.debug("Fixed Label Name: %s" % result.labels.fixed.GetName())
        logging.debug("Moving Image Name: %s" % movingVolume.GetName())
        logging.debug("Moving Label Name: %s" % result.labels.moving.GetName())
        initialTransform = parameterNode.GetAttribute('InitialTransformNodeID')
        if initialTransform:
            initialTransform = slicer.mrmlScene.GetNodeByID(initialTransform)
            logging.debug("Initial Registration Name: %s" % initialTransform.GetName())
        return result

    def run(self, parameterNode, progressCallback=None):
        self.progressCallback = progressCallback
        result = self._processParameterNode(parameterNode)

        registrationTypes = ['rigid', 'affine', 'bSpline']
        self.createVolumeAndTransformNodes(registrationTypes, prefix=str(result.seriesNumber), suffix=result.suffix)

        self.doRigidRegistration(movingBinaryVolume=result.labels.moving, initializeTransformMode="useCenterOfROIAlign")
        self.doAffineRegistration()
        self.doBSplineRegistration(initialTransform=result.transforms.affine)

        targetsNodeID = parameterNode.GetAttribute('TargetsNodeID')
        if targetsNodeID:
            result.targets.original = slicer.mrmlScene.GetNodeByID(targetsNodeID)
            self.transformTargets(registrationTypes, result.targets.original, str(result.seriesNumber), suffix=result.suffix)
        result.volumes.moving = slicer.mrmlScene.GetNodeByID(parameterNode.GetAttribute('MovingImageNodeID'))

    def runReRegistration(self, parameterNode, progressCallback=None):
        logging.debug("Starting Re-Registration")

        self.progressCallback = progressCallback

        self._processParameterNode(parameterNode)
        result = self.registrationResult

        registrationTypes = ['rigid', 'bSpline']
        self.createVolumeAndTransformNodes(registrationTypes, prefix=str(result.seriesNumber), suffix=result.suffix)
        initialTransform = parameterNode.GetAttribute('InitialTransformNodeID')

        if initialTransform:
            initialTransform = slicer.mrmlScene.GetNodeByID(initialTransform)

        self.dilateMask(result.labels.fixed, dilateValue=1)
        self.doRigidRegistration(movingBinaryVolume=result.labels.moving,
                                initialTransform=initialTransform if initialTransform else None)
        self.doBSplineRegistration(initialTransform=result.transforms.rigid, useScaleVersor3D=True, useScaleSkewVersor3D=True,
                                useAffine=True)

        targetsNodeID = parameterNode.GetAttribute('TargetsNodeID')
        if targetsNodeID:
            result.targets.original = slicer.mrmlScene.GetNodeByID(targetsNodeID)
            self.transformTargets(registrationTypes, result.originalTargets, str(result.seriesNumber), suffix=result.suffix)
        result.movingVolume = slicer.mrmlScene.GetNodeByID(parameterNode.GetAttribute('MovingImageNodeID'))

    def createVolumeAndTransformNodes(self, registrationTypes, prefix, suffix=""):
        for regType in registrationTypes:
            self.registrationResult.setVolume(regType, self.createScalarVolumeNode(prefix + '-VOLUME-' + regType + suffix))
            transformName = prefix + '-TRANSFORM-' + regType + suffix
            transform = self.createBSplineTransformNode(transformName) if regType == 'bSpline' \
                else self.createLinearTransformNode(transformName)
            self.registrationResult.setTransform(regType, transform)

    def transformTargets(self, registrations, targets, prefix, suffix=""):
        if targets:
            for registration in registrations:
                name = prefix + '-TARGETS-' + registration + suffix
                clone = self.cloneFiducialAndTransform(name, targets, self.registrationResult.getTransform(registration))
                clone.SetLocked(True)
                self.registrationResult.setTargets(registration, clone)

    def cloneFiducialAndTransform(self, cloneName, originalTargets, transformNode):
        tfmLogic = slicer.modules.transforms.logic()
        clonedTargets = self.cloneFiducials(originalTargets, cloneName)
        clonedTargets.SetAndObserveTransformNodeID(transformNode.GetID())
        tfmLogic.hardenTransform(clonedTargets)
        return clonedTargets

    def doRigidRegistration(self, **kwargs):
        self.updateProgress(labelText='\nRigid registration', value=2)
        paramsRigid = {'fixedVolume': self.registrationResult.volumes.fixed,
                    'movingVolume': self.registrationResult.volumes.moving,
                    'fixedBinaryVolume': self.registrationResult.labels.fixed,
                    'outputTransform': self.registrationResult.transforms.rigid.GetID(),
                    'outputVolume': self.registrationResult.volumes.rigid.GetID(),
                    'maskProcessingMode': "ROI",
                    'useRigid': True}
        for key, value in kwargs.items():
            paramsRigid[key] = value
        slicer.cli.run(slicer.modules.brainsfit, None, paramsRigid, wait_for_completion=True)
        self.registrationResult.cmdArguments += "Rigid Registration Parameters: %s" % str(paramsRigid) + "\n\n"

    def doAffineRegistration(self):
        self.updateProgress(labelText='\nAffine registration', value=2)
        paramsAffine = {'fixedVolume': self.registrationResult.volumes.fixed,
                        'movingVolume': self.registrationResult.volumes.moving,
                        'fixedBinaryVolume': self.registrationResult.labels.fixed,
                        'movingBinaryVolume': self.registrationResult.labels.moving,
                        'outputTransform': self.registrationResult.transforms.affine.GetID(),
                        'outputVolume': self.registrationResult.volumes.affine.GetID(),
                        'maskProcessingMode': "ROI",
                        'useAffine': True,
                        'initialTransform': self.registrationResult.transforms.rigid}
        slicer.cli.run(slicer.modules.brainsfit, None, paramsAffine, wait_for_completion=True)
        self.registrationResult.cmdArguments += "Affine Registration Parameters: %s" % str(paramsAffine) + "\n\n"

    def doBSplineRegistration(self, initialTransform, **kwargs):
        self.updateProgress(labelText='\nBSpline registration', value=3)
        paramsBSpline = {'fixedVolume': self.registrationResult.volumes.fixed,
                        'movingVolume': self.registrationResult.volumes.moving,
                        'outputVolume': self.registrationResult.volumes.bSpline.GetID(),
                        'bsplineTransform': self.registrationResult.transforms.bSpline.GetID(),
                        'fixedBinaryVolume': self.registrationResult.labels.fixed,
                        'movingBinaryVolume': self.registrationResult.labels.moving,
                        'useROIBSpline': True,
                        'useBSpline': True,
                        'splineGridSize': "3,3,3",
                        'maskProcessing': "ROI",
                        'minimumStepLength': "0.005",
                        'maximumStepLength': "0.1",
                        'initialTransform': initialTransform.GetID()}

        for key, value in kwargs.items():
            paramsBSpline[key] = value

        slicer.cli.run(slicer.modules.brainsfit, None, paramsBSpline, wait_for_completion=True)
        self.registrationResult.cmdArguments += "BSpline Registration Parameters: %s" % str(paramsBSpline) + "\n\n"

    def dilateMask(self, maskNode, dilateValue=1):
        dilateErodeParams = {
            "operation": "dilate",
            "inputVolume": maskNode.GetID(),
            "outputVolume": maskNode.GetID(),
            "pixelValue": 1,
            "neighborhood": "CROSS",
            "iterations": dilateValue
        }
        slicer.cli.run(slicer.modules.dilateerode, None, dilateErodeParams, wait_for_completion=True)
        self.registrationResult.cmdArguments += "Dilate mask Parameters: %s" % str(dilateErodeParams) + "\n\n"

    def createBSplineTransformNode(self, name):
        bSplineTransform = slicer.vtkMRMLBSplineTransformNode()
        bSplineTransform.SetName(slicer.mrmlScene.GenerateUniqueName(name))
        slicer.mrmlScene.AddNode(bSplineTransform)
        return bSplineTransform

    def createLinearTransformNode(self, name):
        linearTransform = slicer.vtkMRMLTransformNode()
        linearTransform.SetName(slicer.mrmlScene.GenerateUniqueName(name))
        slicer.mrmlScene.AddNode(linearTransform)
        return linearTransform

    def createScalarVolumeNode(self, name):
        scalarVolume = slicer.vtkMRMLScalarVolumeNode()
        scalarVolume.SetName(slicer.mrmlScene.GenerateUniqueName(name))
        slicer.mrmlScene.AddNode(scalarVolume)
        return scalarVolume

    def cloneFiducials(self, fiducialsNode, cloneName):
        if not fiducialsNode:
            return None
        markupsLogic = slicer.modules.markups.logic()
        originalFiducialList = fiducialsNode.GetID()
        clonedFiducialList = markupsLogic.AddNewFiducialNode(cloneName, slicer.mrmlScene)
        for i in range(fiducialsNode.GetNumberOfFiducials()):
            coordinate = [0, 0, 0]
            fiducialsNode.GetNthFiducialPosition(i, coordinate)
            markupsLogic.AddFiducial(coordinate[0], coordinate[1], coordinate[2], "", clonedFiducialList)
        clonedFiducialList.SetLocked(False)
        return clonedFiducialList

    def updateProgress(self, labelText="", value=None):
        if self.progressCallback:
            self.progressCallback(labelText=labelText, value=value)

class RegistrationResult:
    def __init__(self, seriesNumber):
        self.seriesNumber = seriesNumber
        self.suffix = ""
        self.cmdArguments = ""
        self.volumes = RegistrationResult.Volumes()
        self.labels = RegistrationResult.Labels()
        self.transforms = RegistrationResult.Transforms()
        self.targets = RegistrationResult.Targets()
        self.originalTargets = None

    def setVolume(self, registrationType, volumeNode):
        if registrationType == 'rigid':
            self.volumes.rigid = volumeNode
        elif registrationType == 'affine':
            self.volumes.affine = volumeNode
        elif registrationType == 'bSpline':
            self.volumes.bSpline = volumeNode

    def setTransform(self, registrationType, transformNode):
        if registrationType == 'rigid':
            self.transforms.rigid = transformNode
        elif registrationType == 'affine':
            self.transforms.affine = transformNode
        elif registrationType == 'bSpline':
            self.transforms.bSpline = transformNode

    def setTargets(self, registrationType, targetsNode):
        if registrationType == 'rigid':
            self.targets.rigid = targetsNode
        elif registrationType == 'affine':
            self.targets.affine = targetsNode
        elif registrationType == 'bSpline':
            self.targets.bSpline = targetsNode

    class Volumes:
        def __init__(self):
            self.fixed = None
            self.moving = None
            self.rigid = None
            self.affine = None
            self.bSpline = None

    class Labels:
        def __init__(self):
            self.fixed = None
            self.moving = None

    class Transforms:
        def __init__(self):
            self.rigid = None
            self.affine = None
            self.bSpline = None

    class Targets:
        def __init__(self):
            self.rigid = None
            self.affine = None
            self.bSpline = None

def main(args):
    # Load the volumes and labels into Slicer
    # fixedVolume = slicer.util.getNode(r'C:\Users\pratt\OneDrive\Documents\Sample Images for Sequence Registration\manifest-1720549499368\CPTAC-CM\C3L-00629\04-23-2000-NA-MR BRAIN WOW CONTRAST-18837\203.000000-isoReg - DWI b 1000-03763\1-01.dcm')
    # movingVolume = slicer.util.getNode(r'C:\Users\pratt\Downloads\spl-brain-atlas-master\spl-brain-atlas-master\slicer\volumes\imaging\A1_grayT2.nrrd')
    # fixedLabel = slicer.util.getNode(r'C:\Users\pratt\Downloads\spl-brain-atlas-master\spl-brain-atlas-master\slicer\volumes\labels\hncma-atlas.nrrd')
    # movingLabel = slicer.util.getNode(r'C:\Users\pratt\Downloads\spl-brain-atlas-master\spl-brain-atlas-master\slicer\volumes\labels\hncma-atlas.nrrd')

    fixedVolume = slicer.util.getNode(pattern='203: isoReg - DWI b 1000')
    movingVolume = slicer.util.getNode(pattern='A1_grayT2')
    fixedLabel = slicer.util.getNode(pattern='hncma-atlas-lut')
    movingLabel = slicer.util.getNode(pattern='hncma-atlas-lut')

    # Create and configure the parameter node
    parameterNode = slicer.vtkMRMLScriptedModuleNode()
    parameterNode.SetName("neuroRegistrationParameters")
    parameterNode.SetAttribute('FixedImageNodeID', fixedVolume.GetID())
    parameterNode.SetAttribute('MovingImageNodeID', movingVolume.GetID())
    parameterNode.SetAttribute('FixedLabelNodeID', fixedLabel.GetID())
    parameterNode.SetAttribute('MovingLabelNodeID', movingLabel.GetID())

    # Instantiate and run the logic
    logic = neuroRegistrationLogic()
    logic.run(parameterNode)

# Execute the main function with command-line arguments
main(sys.argv[1:])
