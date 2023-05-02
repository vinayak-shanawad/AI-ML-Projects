import boto3
from datetime import datetime, timedelta
import pandas as pd


def compute_endpoint_metrics(sm_client, cw_client, metric_name, statistics):
    ep_metrics_list = []
    
    # List all endpoints
    response = sm_client.list_endpoints(
        SortBy='Name',
        SortOrder='Ascending',
        MaxResults=100,
        StatusEquals='InService'
    )

    for i in range(len(response["Endpoints"])):
        ep_metrics = {}
        endpoint_name = response["Endpoints"][i]["EndpointName"]
        ep_metrics["Endpoint_Name"] = endpoint_name

        # Retrieve endpoint details
        des_ep_response = sm_client.describe_endpoint(EndpointName=endpoint_name)

        # Retrieve instance type details
        des_epc_response = sm_client.describe_endpoint_config(EndpointConfigName=des_ep_response["EndpointConfigName"])
        if 'InstanceType' in des_epc_response["ProductionVariants"][0]:
            instance_type = des_epc_response["ProductionVariants"][0]["InstanceType"]
            ep_metrics["Instance_Type"] = instance_type
        else:
            continue

        start_time = des_ep_response["CreationTime"]
        end_time = datetime.utcnow()

        # Calculate the days difference between endpoint creation time and present date
        start_date = datetime.date(start_time)
        ep_metrics["Creation_Date"] = start_date

        end_date = datetime.date(end_time)
        days_diff = (end_date - start_date).days

        # Skip the endpoints created on current date
        if days_diff == 0:
            continue

        ep_metrics["Endpoint_Age_In_Days"] = days_diff

        variant_name = des_ep_response["ProductionVariants"][0]["VariantName"]
        ep_metrics["Instance_Count"] = len(des_ep_response["ProductionVariants"])

        period = days_diff * 120

        for k in range(2):
            # Retrieve the cloud watch metric data
            metrics_response = cw_client.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName= metric_name,
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    },
                    {  
                        'Name': 'VariantName', 
                        'Value': variant_name
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=statistics
            )

            # Print the metric data
            datapoints = metrics_response['Datapoints']
            total_invocations = 0
            if datapoints:
                for i in range(len(datapoints)):
                    total_invocations += datapoints[i]['Sum']  # Get the latest data point
            else:
                print("No data available")

            if k == 0:
                ep_metrics["Total_Invocations_Count"] = total_invocations
            else:
                ep_metrics["Invocations_Count_In_Last_15_Days"] = total_invocations

            # Start time for last 15 days
            start_time = datetime.utcnow() - timedelta(hours=360)
            period = 900

        ep_metrics_list.append(ep_metrics
                              )
    return ep_metrics_list
        

# Compute instance cost
def compute_instance_cost(record):    
    if record['Instance_Type'] in instance_cost_details:
        result = float(instance_cost_details[record['Instance_Type']]) * 24 * int(record['Endpoint_Age_In_Days'])
    return result


if __name__ == "__main__":
    
    # Define boto3 clients
    sm_client = boto3.client('sagemaker')
    cw_client = boto3.client('cloudwatch')

    # Define the metric parameters
    metric_name = 'Invocations'
    statistics = ['Sum']

    # Compute endpoint metrics
    ep_metrics_list = compute_endpoint_metrics(sm_client, cw_client, metric_name, statistics)
    
    # Create a endpoint metrics dataframe
    ep_metrics_df = pd.DataFrame(ep_metrics_list)
    
    # Defind the instance cost details per hour
    instance_cost_details = {"ml.g4dn.xlarge": 0.736, "ml.p2.xlarge": 1.125, "ml.c5.2xlarge": 0.408}
    
    ep_metrics_df['Instance_Cost'] = ep_metrics_df.apply(compute_instance_cost, axis=1)

    print(ep_metrics_df.sort_values(by='Total_Invocations_Count', ascending=False, ignore_index=True))