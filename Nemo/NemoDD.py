from getpass import getpass
from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, Field

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


#Initializing config builduer for data designer
config_builder = DataDesignerConfigBuilder(model_configs=model_configs)

config_builder.info.sampler_table



##---------------------------------Product review dataset test---------------------------------------

config_builder.add_column(
    SamplerColumnConfig(
        name="product_category",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=[
                "Electronics",
                "Clothing",
                "Home & Kitchen",
                "Books",
                "Home Office",
            ],
        ),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="product_subcategory",
        sampler_type=SamplerType.SUBCATEGORY,
        params=SubcategorySamplerParams(
            category="product_category",
            values={
                "Electronics": [
                    "Smartphones",
                    "Laptops",
                    "Headphones",
                    "Cameras",
                    "Accessories",
                ],
                "Clothing": [
                    "Men's Clothing",
                    "Women's Clothing",
                    "Winter Coats",
                    "Activewear",
                    "Accessories",
                ],
                "Home & Kitchen": [
                    "Appliances",
                    "Cookware",
                    "Furniture",
                    "Decor",
                    "Organization",
                ],
                "Books": [
                    "Fiction",
                    "Non-Fiction",
                    "Self-Help",
                    "Textbooks",
                    "Classics",
                ],
                "Home Office": [
                    "Desks",
                    "Chairs",
                    "Storage",
                    "Office Supplies",
                    "Lighting",
                ],
            },
        ),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="target_age_range",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["18-25", "25-35", "35-50", "50-65", "65+"]
        ),
    )
)

# Optionally validate that the columns are configured correctly.
config_builder.validate()


#------------------Adding Samplers to generate data related to the customer and their review---------------
# This column will sample synthetic person data based on statistics from the US Census.
config_builder.add_column(
    SamplerColumnConfig(
        name="customer",
        sampler_type=SamplerType.PERSON,
        params=PersonSamplerParams(age_range=[18, 70]),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="number_of_stars",
        sampler_type=SamplerType.UNIFORM,
        params=UniformSamplerParams(low=1, high=5),
        convert_to="int",  # Convert the sampled float to an integer.
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="review_style",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["rambling", "brief", "detailed", "structured with bullet points"],
            weights=[1, 2, 2, 1],
        ),
    )
)

config_builder.validate()

#------------------------------LLM GENERATED COLUMNS-----------------------------------
config_builder.add_column(
    LLMTextColumnConfig(
        name="product_name",
        prompt=(
            "You are a helpful assistant that generates product names. DO NOT add quotes around the product name.\n\n"
            "Come up with a creative product name for a product in the '{{ product_category }}' category, focusing "
            "on products related to '{{ product_subcategory }}'. The target age range of the ideal customer is "
            "{{ target_age_range }} years old. Respond with only the product name, no other text."
        ),
        system_prompt=SYSTEM_PROMPT,
        model_alias=MODEL_ALIAS,
    )
)

config_builder.add_column(
    LLMTextColumnConfig(
        name="customer_review",
        prompt=(
            "You are a customer named {{ customer.first_name }} from {{ customer.city }}, {{ customer.state }}. "
            "You are {{ customer.age }} years old and recently purchased a product called {{ product_name }}. "
            "Write a review of this product, which you gave a rating of {{ number_of_stars }} stars. "
            "The style of the review should be '{{ review_style }}'."
        ),
        system_prompt=SYSTEM_PROMPT,
        model_alias=MODEL_ALIAS,
    )
)

config_builder.validate()




#--------------------Testing for preview of a sample record----------------


# preview = data_designer_client.preview(config_builder)



# # Run this cell multiple times to cycle through the 10 preview records.
# for i in range(0,10):
#     preview.display_sample_record()

# # The preview dataset is available as a pandas DataFrame.
# # TO VIEW THE DATA FRAME, RUN THIS IN DEBUGGER AND USE THE DATA WRANGLER EXTENSION IN VSC
# preview.dataset

# # Print the analysis as a table.
# preview.analysis.to_report()


#------------------------------------FINALLY CREATE THE DATASET-----------------------------------
job_results = data_designer_client.create(config_builder, num_records=20)

# This will block until the job is complete.
job_results.wait_until_done()



# Load the generated dataset as a pandas DataFrame.
dataset = job_results.load_dataset()

dataset.head()



# Load the analysis results into memory.
analysis = job_results.load_analysis()

analysis.to_report()


OUTPUT_PATH = "GenDatasets"

# Download the job artifacts and save them to disk.
job_results.download_artifacts(
    output_path=OUTPUT_PATH,
    artifacts_folder_name="artifacts-1-the-basics",
);