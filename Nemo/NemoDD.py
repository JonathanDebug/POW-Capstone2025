from getpass import getpass
from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, Field
from pathlib import Path


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
    LLMStructuredColumnConfig,
    ExpressionColumnConfig
)


##INITIALIZING Data_path
project_path = Path(__file__).parent.parent
data_path = project_path / "data" 


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
            temperature=0.8,
            top_p=1.0,
            max_tokens=1024,
        ),
    )
]


#Initializing config builduer for data designer
config_builder = DataDesignerConfigBuilder(model_configs=model_configs)

config_builder.info.sampler_table


# Configuring product Schema

# We define a Product schema so that the name, description, and price are generated
# in one go, with the types and constraints specified.
# class Product(BaseModel):
#     name: str = Field(description="The name of the product")
#     description: str = Field(description="A description of the product")
#     price: Decimal = Field(
#         description="The price of the product", ge=10, le=1000, decimal_places=2
#     )


# class ProductReview(BaseModel):
#     rating: int = Field(description="The rating of the product", ge=1, le=5)
#     customer_mood: Literal["irritated", "mad", "happy", "neutral", "excited"] = Field(
#         description="The mood of the customer"
#     )
#     review: str = Field(description="A review of the product")

#Defining a Student Object to be created in one go
# class Student(BaseModel):
#     name: str = Field(description="Full name of student"),
#     age: int = Field(description="student age" ge=18,ge=30)


##---------------------------------Product review dataset test---------------------------------------

# Since we often only want a few attributes from Person objects, we can
# set drop=True in the column config to drop the column from the final dataset.

#We're generating even some columns that their purpose is to help generate the proper subject and email
#Consider it like helper columns
config_builder.add_column(
    SamplerColumnConfig(
        name="student_victim", #That's what they are after all
        sampler_type=SamplerType.PERSON,
        params=PersonSamplerParams(locale="en_US", age_range=[18, 30], with_synthetic_personas=False),
        drop=True
         # Range ofage of students possibly vulnerable
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="departments",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=[
                "Computer Science and Engineering",
                "Electrical and Computer Engineering",
                "Mechanical Engineering",
                "Industrial Engineering",
                "Chemical Engineering",
            ],
        ),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="department_staff",
        sampler_type=SamplerType.SUBCATEGORY,
        params=SubcategorySamplerParams(
            category="departments",
            values={
                "Computer Science and Engineering": [
                    "Dr. William Tercero Gallina",
                    "Dr. Kaige Lu",
                    "Dr. Mark Shoots",
                    "Dr. Juan Midely",
                    "Dr. Juan Patrollo",
                ],
                "Electrical and Computer Engineering": [
                    "Dr. Easydoro Rey",
                    "Dr. Gather Shun",
                    "Dr. Edward Otto",
                    "Dr. Cordova Bonilla",
                    "Dr. Domingo Domingo Rodriguez",
                ],
                "Mechanical Engineering": [
                    "Dr. Sierra David",
                    "Dr. Ludwig Jose",
                    "Dr. Brand Coolriel",
                    "Dr. David Doomer",
                    "Dr. Pedro Pedro",
                ],
                "Industrial Engineering": [
                    "Dr. Agosto Rulla Toro",
                    "Dr. Hector Carlos Carlos",
                    "Dr. David Barrio Gonzales",
                    "Assistant Professor Sam Olive Bonicia",
                    "Dr. Mauricio Rivera Dominguez",
                ],
                "Chemical Engineering": [
                    "Dra. Marie Curie",
                    "Dr. Yoma Torres",
                    "Dra. Patricia Bermuda Otto",
                    "Dr. Osvaldo Cordova",
                    "Dr. Arturo Rey",
                ],
            },
        ),
    )
)


# Sampler columns support conditional params, which are used if the condition is met.
# In this example, we set the review style to rambling if the target age range is 18-25.
# Note conditional parameters are only supported for Sampler column types.
config_builder.add_column(
    SamplerColumnConfig(
        name="subject",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["remote research opportunity", "research opportunity", "research assistant","administrative assistant", "Internship"],
            weights=[2,1,2,1,1],
        ),

    )
)

# Optionally validate that the columns are configured correctly.
config_builder.validate()


#------------------Adding Samplers to generate data related to the customer and their review---------------
# This column will sample synthetic person data based on statistics from the US Census.
# config_builder.add_column(
#     SamplerColumnConfig(
#         name="customer",
#         sampler_type=SamplerType.PERSON,
#         params=PersonSamplerParams(age_range=[18, 65]),
#         drop=True,
#     )
# )

# config_builder.add_column(
#     SamplerColumnConfig(
#         name="number_of_stars",
#         sampler_type=SamplerType.UNIFORM,
#         params=UniformSamplerParams(low=1, high=5),
#         convert_to="int",  # Convert the sampled float to an integer.
#     )
# )

# config_builder.add_column(
#     SamplerColumnConfig(
#         name="review_style",
#         sampler_type=SamplerType.CATEGORY,
#         params=CategorySamplerParams(
#             values=["rambling", "brief", "detailed", "structured with bullet points"],
#             weights=[1, 2, 2, 1],
#         ),
#     )
# )

# config_builder.validate()

#------------------------------LLM GENERATED COLUMNS-----------------------------------
# We can create new columns using Jinja expressions that reference
# existing columns, including attributes of nested objects.
config_builder.add_column(
    ExpressionColumnConfig(
        name="student_name", expr="{{ student_victim.first_name }} {{ student_victim.last_name }}"
    )
)
config_builder.add_column(
    ExpressionColumnConfig(
        name="student_age", expr="{{ student_victim.age }}",
        dtype="int",
    )
)

# config_builder.add_column(
#     ExpressionColumnConfig(name="customer_age", expr="{{ customer.age }}")
# )


config_builder.add_column(
    LLMTextColumnConfig(
        name="email_subject",
        prompt=(
                "Create an email subject from '{{subject}}' category, offering a position relating " 
                "to the university of Puerto Rico's '{{departments}}'. The subject should not be longer than 1 sentence. " 
                "The subject should be direct and concise about the position offered."
                ),
        system_prompt=SYSTEM_PROMPT,
        model_alias=MODEL_ALIAS,
    )
)

config_builder.add_column(
    LLMTextColumnConfig(
        name="body",
        prompt=(
                "Create an email body text that offers a job position from the '{{subject}}' category, related to the University of Puerto Rico " 
                "at Mayaguez '{{departments}}' department directed to the Students of '{{departments}}'. The email body text provides a description of the job position displayed in a list. "
                "Keep the job description vague and short. " 
                "The email body text should list the requirements needed for the job position. "
                "The requirements must have a low barrier of entry for the students in the '{{departments}}' field. "
                "The email should have the weekly pay listed. With the weekly pay being priced between 250$ and 350$. It must show a singular value. "
                "The email should be signed off with the name of the '{{department_staff}}'" 
                "{% if student_age < 22 %}"
                "Address the student directly by '{{student_name}}' "
                "Change the requirements to say that there's no prior experience needed for the job "
                "{% else %}"
                "Address the student body of the '{{departments}}' as a whole. "
                "{% endif %}"
                ),
                system_prompt=SYSTEM_PROMPT,
                model_alias=MODEL_ALIAS,
    )
)

# config_builder.add_column(
#     LLMStructuredColumnConfig(
#         name="body",
#         prompt=(
#             "Create a product in the '{{ product_category }}' category, focusing on products  "
#             "related to '{{ product_subcategory }}'. The target age range of the ideal customer is "
#             "{{ target_age_range }} years old. The product should be priced between $10 and $1000."
#         ),
#         system_prompt=SYSTEM_PROMPT,
#         output_format=Product,
#         model_alias=MODEL_ALIAS,
#     )
# )

# We can even use if/else logic in our Jinja expressions to create more complex prompt patterns.
# config_builder.add_column(
#     LLMStructuredColumnConfig(
#         name="customer_review",
#         prompt=(
#             "Your task is to write a review for the following product:\n\n"
#             "Product Name: {{ product.name }}\n"
#             "Product Description: {{ product.description }}\n"
#             "Price: {{ product.price }}\n\n"
#             "Imagine your name is {{ customer_name }} and you are from {{ customer.city }}, {{ customer.state }}. "
#             "Write the review in a style that is '{{ review_style }}'."
#             "{% if target_age_range == '18-25' %}"
#             "Make sure the review is more informal and conversational."
#             "{% else %}"
#             "Make sure the review is more formal and structured."
#             "{% endif %}"
#         ),
#         system_prompt=SYSTEM_PROMPT,
#         output_format=ProductReview,
#         model_alias=MODEL_ALIAS,
#     )
# )

config_builder.validate()




#--------------------Testing for preview of a sample record----------------


preview = data_designer_client.preview(config_builder)



# Run this cell multiple times to cycle through the 10 preview records.
for i in range(0,10):
    preview.display_sample_record()

# The preview dataset is available as a pandas DataFrame.
# TO VIEW THE DATA FRAME, RUN THIS IN DEBUGGER AND USE THE DATA WRANGLER EXTENSION IN VSC

preview.dataset

# Print the analysis as a table.
preview.analysis.to_report()


#------------------------------------FINALLY CREATE THE DATASET-----------------------------------
# job_results = data_designer_client.create(config_builder, num_records=100)

# # This will block until the job is complete.
# job_results.wait_until_done()



# # Load the generated dataset as a pandas DataFrame.
# dataset = job_results.load_dataset()

# dataset.head()



# # Load the analysis results into memory.
# analysis = job_results.load_analysis()

# analysis.to_report()


# OUTPUT_PATH = "GenDatasets"

# # Download the job artifacts and save them to disk.
# job_results.download_artifacts(
#     output_path=OUTPUT_PATH,
#     artifacts_folder_name="artifacts-2-structured-outputs-and-jinja-expressions",
# );