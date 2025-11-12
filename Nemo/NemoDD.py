from getpass import getpass

from nemo_microservices.data_designer.essentials import (
    CategorySamplerParams,
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    NeMoDataDesignerClient,
    PersonSamplerParams,
    SamplerColumnConfig,
    SamplerType,
    SubcategorySamplerParams,
    UniformSamplerParams,
    ModelConfig,
    InferenceParameters,
)


#Initializing the NeMo Data Designer Client
NEMO_MICROSERVICES_BASE_URL = "http://localhost:8080"

data_designer_client = NeMoDataDesignerClient(base_url=NEMO_MICROSERVICES_BASE_URL)


#Define MODEL CONFIGURATION ------------------------------------------------------

# This name is set in the microservice deployment configuration.
MODEL_PROVIDER = "nvidiabuild"

# The model ID is from build.nvidia.com.
MODEL_ID = "nvidia/nvidia-nemotron-nano-9b-v2"

# We choose this alias to be descriptive for our use case.
MODEL_ALIAS = "nemotron-nano-v2"

# This sets reasoning to False for the nemotron-nano-v2 model.
SYSTEM_PROMPT = "/no_think"

model_configs = [
    ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=InferenceParameters(
            temperature=0.5,
            top_p=1.0,
            max_tokens=1024,
        ),
    )
]