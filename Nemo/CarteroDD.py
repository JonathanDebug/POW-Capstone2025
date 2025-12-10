from getpass import getpass
from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List


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
MODEL_ALIAS = "nemotronDDPhishDataset"

# This sets reasoning to False for the nemotron-nano-v2 model.
SYSTEM_PROMPT = "/no_think"


model_configs = [
    ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=InferenceParameters(
            temperature=0.9,
            top_p=0.9,
            max_tokens=1024,
            max_parallel_requests=4,
            timeout=45.
        ),
    )
]


#Initializing config builduer for data designer
config_builder = DataDesignerConfigBuilder(model_configs=model_configs)

config_builder.info.sampler_table


# Configuring product Schema

# class Email(BaseModel):
#     subject: str = Field(description="The Email's Subject to grap the student's attention")
#     text: str = Field(description="Email text body showcasing the entire email")

# class Event(BaseModel):
#   event_title: str = Field(description= "Title of university event")
#   location: str = Field(description="Where the even will be hosted, limited to areas within a University")
#   date: str = Field(description="Date in the format of D/M/YYYY")
#   time: str = Field(description="time of the event")
#   description: str = Field(description="brief event description")

  





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
        name="Activity_topics",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=[
                "Academics & Research",
                "Campus Life & Events",
                "Local Opportunities & Engagement",
                "Logistics & Operations",
                "Health, Safety & Wellness",
                "Puerto Rico Specific",
                "Featured Section",
            ],
        ),
    )
)

config_builder.add_column(
    SamplerColumnConfig(
        name="Activity",
        sampler_type=SamplerType.SUBCATEGORY,
        params=SubcategorySamplerParams(
            category="Activity_topics",
            values={
                "Academics & Research": [
                    "Deadline Alerts",
                    "Workshop & Seminar Highlights",
                    "Research Opportunities",
                    "Faculty Spotlight",
                ],
                "Campus Life & Events": [
                    "Today's Events",
                    "Student Org Spotlight",
                    "Arts & Culture",
                    "Athletics & Recreation",
                ],
                "Local Opportunities & Engagement": [
                    "Internships & Jobs",
                    "Community Service",
                    "Cultural Events in PR",
                    "Local Business Highlights",
                ],
                "Logistics & Operations": [
                    "Campus Announcements",
                    "Transportation",
                    "Facility Hours",
                ],
                "Health, Safety & Wellness": [
                    "Health Services",
                    "Safety Updates",
                    "Wellness Tips",
                ],
                "Puerto Rico Specific": [
                  "Weather & Preparedness", 
                  "Heritage & History",
                  "Sustainability & Environment",
                  "Language & Culture",
                ],
                "Featured Section": [
                  "Conoce tu Isla",
                  "Student of the Day/Week",
                  "Did you know? Fun Facts about Puerto Rico"
                ]
            },
        ),
    )
)



# Optionally validate that the columns are configured correctly.
config_builder.validate()


#------------------Adding Samplers to generate data related to the customer and their review---------------

#------------------------------LLM GENERATED COLUMNS-----------------------------------
# We can create new columns using Jinja expressions that reference
# existing columns, including attributes of nested objects.
config_builder.add_column(
    SamplerColumnConfig(
        name="num_events",
        sampler_type=SamplerType.UNIFORM,
        params=UniformSamplerParams(low=4, high=7),
        convert_to="int"  # Optional: converts to integer
    )
)



# config_builder.add_column(
#     ExpressionColumnConfig(name="customer_age", expr="{{ customer.age }}")
# )
# config_builder.add_column(
#     LLMStructuredColumnConfig(
#         name="university_event",
#         prompt=(
#                 "Create a campus event based on '{{Activity}}' "
#                 "Each event needs to have an event title, location, date/time, "
#                 "and a short sentence for a description. "
#                 "Each Event should be given in the following format: "
#                 "event_title: Title of university event"
#                 "location: Where the even will be hosted, limited to areas within the University of Puerto Rico"
#                 "date: Date in the format of D/M/YYYY"
#                 "time: time of the event"
#                 "description: brief event description"
#                 ),
#         system_prompt=SYSTEM_PROMPT,
#         output_format=Event,
#         model_alias=MODEL_ALIAS,
#     )
# )



config_builder.add_column(
    LLMTextColumnConfig(
        name="email_subject",
        prompt=(
                "Write an email subject for a University Daily Digests(\"CARTERO\") "
                "that looks to grab the attention of students from the University of Puerto Rico at Mayaguez "
                "Highlight the main '{{Activity}}' "
                "Keep it 1 sentence max. "
                ),
        system_prompt=SYSTEM_PROMPT,
        model_alias=MODEL_ALIAS,
    )
)

config_builder.add_column(
    LLMTextColumnConfig(
        name="body",
        prompt=(
                "Write a University Daily digest email for the University of Puerto Rico at Mayaguez named CARTERO "
                "Begin with the university name and a brief greeting. "
                "Create a list of '{{num_events}}' relating to the topic of '{{Activity}}'"
                "Each Event should have it's own individual event title, location, date, time, and a short sentence describint it."
                "The Location should be limited to the campus for the University of Puerto Rico at Mayaguez."
                "The Date should be varied all year long with the format of Day of Month,Month, Year"
                "Keep the tone friendly, informative, and appropriate for a student audience. "
                "End with a short \"About Cartero\" section explaining that the email is an automated digest of all campus events and a contact so that people can put up their own announcements."
                ),
                system_prompt=SYSTEM_PROMPT,
                model_alias=MODEL_ALIAS,
    )
)


config_builder.add_column(
    SamplerColumnConfig(
        name="label",
        sampler_type=SamplerType.UNIFORM,
        params=UniformSamplerParams(low=0, high=0),
        convert_to="int"  # Optional: converts to integer
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
job_results = data_designer_client.create(config_builder, num_records=25000)

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
    artifacts_folder_name="Phishing_Emails_test2",
);
