# contains information on e.g. which example-names are valid, etc.

validExampleNames = ["arap_mesh_deformation",
        "cotangent_mesh_smoothing",
        "embedded_mesh_deformation",
        "image_warping",
        "intrinsic_image_decomposition",
        "optical_flow",
        "poisson_image_editing",
        "robust_nonrigid_alignment",
        "shape_from_shading",
        "volumetric_mesh_deformation"]


validExampleCeres = ["arap_mesh_deformation",
        "image_warping",
        "poisson_image_editing",
        "shape_from_shading",
        "volumentric_mesh_deformation"]


#------------------------------------------------------------------------------
# names for which we want to parse timings from opt. for example, in image
# warping we need 'PCGStep1-3' to get linear iteration timings. The names
# * will be different for each materialization strategy.
# * will only work for cpu backend
# * are used by the outParse module

# first comes example-specific stuff and at the end we add whatever
# is common to all examples


optPerSolveNames = {}
optPerSolveNames['matfree'] = {}
optPerSolveNames['JTJ'] = {}
optPerSolveNames['fusedJTJ'] = {}

optPerNewtonNames = {}
optPerNewtonNames['matfree'] = {}
optPerNewtonNames['JTJ'] = {}
optPerNewtonNames['fusedJTJ'] = {}

optPerLinIterNames = {}
optPerLinIterNames['matfree'] = {}
optPerLinIterNames['JTJ'] = {}
optPerLinIterNames['fusedJTJ'] = {}



optPerSolveNames['matfree']['arap_mesh_deformation'] = []
optPerNewtonNames['matfree']['arap_mesh_deformation'] = ['PCGInit1_Graph', 'PCGInit1_Finish', 'computeCost_Graph']
optPerLinIterNames['matfree']['arap_mesh_deformation'] = []

optPerSolveNames['matfree']['cotangent_mesh_smoothing'] = []
optPerNewtonNames['matfree']['cotangent_mesh_smoothing'] = ['PCGInit1_Graph', 'PCGInit1_Finish', 'computeCost_Graph']
optPerLinIterNames['matfree']['cotangent_mesh_smoothing'] = []

optPerSolveNames['matfree']['embedded_mesh_deformation'] = ['PCGSaveSSq']
optPerNewtonNames['matfree']['embedded_mesh_deformation'] = ['PCGInit1_Graph', 'PCGInit1_Finish', 'computeCost_Graph',
        'PCGComputeCtC', 'PCGComputeCtC_Graph', 'PCGFinalizeDiagonal', 'computeModelCost', 'computeModelCost_Graph',
        'savePreviousUnknowns']
optPerLinIterNames['matfree']['embedded_mesh_deformation'] = []

optPerSolveNames['matfree']['image_warping'] = []
optPerNewtonNames['matfree']['image_warping'] = []
optPerLinIterNames['matfree']['image_warping'] = []

optPerSolveNames['matfree']['intrinsic_image_decomposition'] = []
optPerNewtonNames['matfree']['intrinsic_image_decomposition'] = ['precompute']
optPerLinIterNames['matfree']['intrinsic_image_decomposition'] = []

optPerSolveNames['matfree']['optical_flow'] = []
optPerNewtonNames['matfree']['optical_flow'] = []
optPerLinIterNames['matfree']['optical_flow'] = []

optPerSolveNames['matfree']['poisson_image_editing'] = []
optPerNewtonNames['matfree']['poisson_image_editing'] = []
optPerLinIterNames['matfree']['poisson_image_editing'] = []

# TODO this example is currently broken
optPerSolveNames['matfree']['robust_nonrigid_alignment'] = []
optPerNewtonNames['matfree']['robust_nonrigid_alignment'] = []
optPerLinIterNames['matfree']['robust_nonrigid_alignment'] = []

optPerSolveNames['matfree']['shape_from_shading'] = []
optPerNewtonNames['matfree']['shape_from_shading'] = ['precompute']
optPerLinIterNames['matfree']['shape_from_shading'] = []

optPerSolveNames['matfree']['volumetric_mesh_deformation'] = []
optPerNewtonNames['matfree']['volumetric_mesh_deformation'] = []
optPerLinIterNames['matfree']['volumetric_mesh_deformation'] = []







optPerSolveNames['JTJ']['arap_mesh_deformation'] = []
optPerNewtonNames['JTJ']['arap_mesh_deformation'] = ['PCGInit1_Graph', 'PCGInit1_Finish', 'computeCost_Graph', 'saveJToCRS_Graph']
optPerLinIterNames['JTJ']['arap_mesh_deformation'] = []

optPerSolveNames['JTJ']['cotangent_mesh_smoothing'] = []
optPerNewtonNames['JTJ']['cotangent_mesh_smoothing'] = ['PCGInit1_Graph', 'PCGInit1_Finish', 'computeCost_Graph', 'saveJToCRS_Graph']
optPerLinIterNames['JTJ']['cotangent_mesh_smoothing'] = []

optPerSolveNames['JTJ']['embedded_mesh_deformation'] = ['PCGSaveSSq']
optPerNewtonNames['JTJ']['embedded_mesh_deformation'] = ['PCGInit1_Graph', 'PCGInit1_Finish', 'computeCost_Graph',
        'PCGComputeCtC', 'PCGComputeCtC_Graph', 'PCGFinalizeDiagonal', 'computeModelCost', 'computeModelCost_Graph',
        'savePreviousUnknowns', 'saveJToCRS_Graph']
optPerLinIterNames['JTJ']['embedded_mesh_deformation'] = []

optPerSolveNames['JTJ']['image_warping'] = []
optPerNewtonNames['JTJ']['image_warping'] = []
optPerLinIterNames['JTJ']['image_warping'] = []

optPerSolveNames['JTJ']['intrinsic_image_decomposition'] = []
optPerNewtonNames['JTJ']['intrinsic_image_decomposition'] = ['precompute']
optPerLinIterNames['JTJ']['intrinsic_image_decomposition'] = []

optPerSolveNames['JTJ']['optical_flow'] = []
optPerNewtonNames['JTJ']['optical_flow'] = []
optPerLinIterNames['JTJ']['optical_flow'] = []

optPerSolveNames['JTJ']['poisson_image_editing'] = []
optPerNewtonNames['JTJ']['poisson_image_editing'] = []
optPerLinIterNames['JTJ']['poisson_image_editing'] = []

# TODO this example is currently broken
optPerSolveNames['JTJ']['robust_nonrigid_alignment'] = []
optPerNewtonNames['JTJ']['robust_nonrigid_alignment'] = []
optPerLinIterNames['JTJ']['robust_nonrigid_alignment'] = []

optPerSolveNames['JTJ']['shape_from_shading'] = []
optPerNewtonNames['JTJ']['shape_from_shading'] = ['precompute']
optPerLinIterNames['JTJ']['shape_from_shading'] = []

optPerSolveNames['JTJ']['volumetric_mesh_deformation'] = []
optPerNewtonNames['JTJ']['volumetric_mesh_deformation'] = []
optPerLinIterNames['JTJ']['volumetric_mesh_deformation'] = []






optPerSolveNames['fusedJTJ']['arap_mesh_deformation'] = []
optPerNewtonNames['fusedJTJ']['arap_mesh_deformation'] = [ 'PCGInit1_Graph', 'PCGInit1_Finish', 'computeCost_Graph', 'saveJToCRS_Graph']
optPerLinIterNames['fusedJTJ']['arap_mesh_deformation'] = []

optPerSolveNames['fusedJTJ']['cotangent_mesh_smoothing'] = []
optPerNewtonNames['fusedJTJ']['cotangent_mesh_smoothing'] = [ 'PCGInit1_Graph', 'PCGInit1_Finish', 'computeCost_Graph', 'saveJToCRS_Graph']
optPerLinIterNames['fusedJTJ']['cotangent_mesh_smoothing'] = []

optPerSolveNames['fusedJTJ']['embedded_mesh_deformation'] = ['PCGSaveSSq']
optPerNewtonNames['fusedJTJ']['embedded_mesh_deformation'] = ['PCGInit1_Graph', 'PCGInit1_Finish', 'computeCost_Graph',
        'PCGComputeCtC', 'PCGComputeCtC_Graph', 'PCGFinalizeDiagonal', 'computeModelCost', 'computeModelCost_Graph',
        'savePreviousUnknowns', 'saveJToCRS_Graph']
optPerLinIterNames['fusedJTJ']['embedded_mesh_deformation'] = []

optPerSolveNames['fusedJTJ']['image_warping'] = []
optPerNewtonNames['fusedJTJ']['image_warping'] = []
optPerLinIterNames['fusedJTJ']['image_warping'] = []

optPerSolveNames['fusedJTJ']['intrinsic_image_decomposition'] = []
optPerNewtonNames['fusedJTJ']['intrinsic_image_decomposition'] = ['precompute']
optPerLinIterNames['fusedJTJ']['intrinsic_image_decomposition'] = []

optPerSolveNames['fusedJTJ']['optical_flow'] = []
optPerNewtonNames['fusedJTJ']['optical_flow'] = []
optPerLinIterNames['fusedJTJ']['optical_flow'] = []

optPerSolveNames['fusedJTJ']['poisson_image_editing'] = []
optPerNewtonNames['fusedJTJ']['poisson_image_editing'] = []
optPerLinIterNames['fusedJTJ']['poisson_image_editing'] = []

# TODO this example is currently broken
optPerSolveNames['fusedJTJ']['robust_nonrigid_alignment'] = []
optPerNewtonNames['fusedJTJ']['robust_nonrigid_alignment'] = []
optPerLinIterNames['fusedJTJ']['robust_nonrigid_alignment'] = []

optPerSolveNames['fusedJTJ']['shape_from_shading'] = []
optPerNewtonNames['fusedJTJ']['shape_from_shading'] = ['precompute']
optPerLinIterNames['fusedJTJ']['shape_from_shading'] = []

optPerSolveNames['fusedJTJ']['volumetric_mesh_deformation'] = []
optPerNewtonNames['fusedJTJ']['volumetric_mesh_deformation'] = []
optPerLinIterNames['fusedJTJ']['volumetric_mesh_deformation'] = []


# stuff that is common to all examples
for ex_name in validExampleNames:
    optPerSolveNames['matfree'][ex_name] += []
    optPerNewtonNames['matfree'][ex_name] += ['PCGInit1', 'computeCost', 'PCGLinearUpdate']
    optPerLinIterNames['matfree'][ex_name] += ['linear iteration']

    optPerSolveNames['JTJ'][ex_name] += ['JT alloc']
    optPerNewtonNames['JTJ'][ex_name] += ['PCGInit1', 'computeCost', 'PCGLinearUpdate', 'saveJToCRS', 'J_transpose']
    optPerLinIterNames['JTJ'][ex_name] += ['linear iteration']

    # NOTE: we need a backslash to escape the ^ becuase we want to
    # use the name in a regex
    optPerSolveNames['fusedJTJ'][ex_name] += ['JT alloc', 'J\^TJ alloc']
    optPerNewtonNames['fusedJTJ'][ex_name] += ['PCGInit1', 'computeCost', 'PCGLinearUpdate', 'saveJToCRS', 'J_transpose', 'JTJ multiply']
    optPerLinIterNames['fusedJTJ'][ex_name] += ['linear iteration']


#------------------------------------------------------------------------------
# the flags for pixelstuff and graphstuff are different, see ArgParser.h.
strideFlags = {}
strideFlags['arap_mesh_deformation'] = "--numSubdivides"
strideFlags['cotangent_mesh_smoothing'] = "--numSubdivides"
strideFlags['embedded_mesh_deformation'] = "--numSubdivides"
strideFlags['image_warping'] = "--stride"
strideFlags['intrinsic_image_decomposition'] = "--stride"
strideFlags['optical_flow'] = "--stride"
strideFlags['poisson_image_editing'] = "--stride"
strideFlags['robust_nonrigid_alignment'] = "--numSubdivides"
strideFlags['shape_from_shading'] = "--numSubdivides" # TODO
strideFlags['volumetric_mesh_deformation'] = "--numSubdivides"
