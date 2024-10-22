add_triton_library(TritonCPUTransforms
    Canonicalize.cpp
    ConvertDotGeneric.cpp
    ConvertDotProduct.cpp
    ConvertDotToAMX.cpp
    ConvertUnsupportedOps.cpp
    DecomposeFpConversions.cpp
    OptimizeMasks.cpp

    DEPENDS
    TritonCPUTransformsPassIncGen
)
