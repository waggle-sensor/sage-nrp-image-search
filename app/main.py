'''This file contains the code to generate the gradio app'''
#NOTE: This is will be replaced with our UI in k8s namespace beekeeper.
#   The UI will use the new queries added to our data API to do the same thing this file is doing.

import gradio as gr
import os
import weaviate
from weaviate.classes.init import Timeout, AdditionalConfig
import argparse
import logging
import time
import tritonclient.grpc as TritonClient
import plotly.graph_objects as go
import pandas as pd
from query import Weav_query, Sage_query

# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','jfif'])
IMAGE_DIR = os.path.join(os.getcwd(), "static", "Images")
UPLOAD_DIR = os.path.join(os.getcwd(), "static", "uploads")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)

def allowed_file(filename):
    '''
    Check if file is allowed
    '''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_weaviate_client():
    '''
    Intialize weaviate client based on arg or env var
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weaviate_host",
        default=os.getenv("WEAVIATE_HOST","127.0.0.1"),
        help="Weaviate host IP.",
    )
    parser.add_argument(
        "--weaviate_port",
        default=os.getenv("WEAVIATE_PORT","8080"),
        help="Weaviate REST port.",
    )
    parser.add_argument(
        "--weaviate_grpc_port",
        default=os.getenv("WEAVIATE_GRPC_PORT","50051"),
        help="Weaviate GRPC port.",
    )
    args = parser.parse_args()

    weaviate_host = args.weaviate_host
    weaviate_port = args.weaviate_port
    weaviate_grpc_port = args.weaviate_grpc_port

    logging.debug(f"Attempting to connect to Weaviate at {weaviate_host}:{weaviate_port}")

    # Retry logic to connect to Weaviate
    while True:
        try:
            add_config = AdditionalConfig(
                timeout=Timeout(init=15, query=120, insert=120)
            )
            client = weaviate.connect_to_local(
                host=weaviate_host,
                port=weaviate_port,
                grpc_port=weaviate_grpc_port,
                additional_config=add_config,
            )
            logging.debug("Successfully connected to Weaviate")
            return client
        except weaviate.exceptions.WeaviateConnectionError as e:
            logging.error(f"Failed to connect to Weaviate: {e}")
            logging.debug("Retrying in 10 seconds...")
            time.sleep(10)

# TODO: implement testImage() first
# def image_query(file):
#     '''
#     Send image query to testImage() and engineer results to display in Gradio
#     '''
#     # Get the full path of the temporary file
#     temp_file_path = file.name
    
#     # Define the destination file path
#     full_path = os.path.join(UPLOAD_DIR, os.path.basename(temp_file_path))
    
#     # Move the temporary file to the destination directory
#     shutil.move(temp_file_path, full_path)
    
#     dic = testImage({"image": f"{full_path}"}, client)
#     image_paths = dic['objects']
#     certainty = dic['scores']

#     images = [f'{IMAGE_DIR}/{i}' for i in image_paths if any(ext in i for ext in ['.jfif','.jpg','.jpeg','.png'])]

#     return images, certainty

def filter_map(df):
    '''
    This function generates a map based on the results from text_query
    '''
    #set up custom data
    uuid = df["uuid"].tolist()
    address = df["address"].tolist() 
    score = df["score"].tolist()
    rerank_score = df["rerank_score"].tolist()
    data = [(uuid[i], address[i], score[i], rerank_score[i]) for i in range(0, len(uuid))]

    # Create the plot
    fig = go.Figure(go.Scattermapbox(
        lat=df['location_lat'].tolist(),
        lon=df['location_lon'].tolist(),
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=6
        ),
        hoverinfo="text",
        hovertemplate='<b>uuid</b>: %{customdata[0]}<br><b>address</b>: %{customdata[1]}<br><b>score</b>: %{customdata[2]}<br><b>rerank_score</b>: %{customdata[3]}',
        customdata=data
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=39.00,
                lon=-98.00
            ),
            pitch=0,
            zoom=1
        ),
    )

    return fig

weaviate_client = initialize_weaviate_client()

TRITON_HOST = os.environ.get("TRITON_HOST","triton")
TRITON_PORT = os.environ.get("TRITON_PORT","8001")
triton_client = TritonClient.InferenceServerClient(url=f"{TRITON_HOST}:{TRITON_PORT}")

wq = Weav_query(weaviate_client, triton_client)
sq = Sage_query()
def text_query(description):
    '''
    Send text query to a weaviate query and engineer results to display in Gradio
    '''
    # send the query to Weaviate and get the results
    df = wq.clip_hybrid_query(description)

    # authorize results based on allowed nodes
    # TODO: implement auth using username and key from sage user
    df = df[df['vsn'].apply(lambda x: sq.authorize(x))]

    # Extract the image links and captions from the DataFrame
    images = []
    for _, row in df.iterrows():  # Iterate through the DataFrame rows
        if any(row["filename"].endswith(ext) for ext in [".jfif", ".jpg", ".jpeg", ".png"]):
            # Use getImage to retrieve the image from the URL
            image = sq.getImage(row['link'])
            if image:
                images.append((image, f"{row['uuid']}"))

    #get location details
    location = df[['location_lat', 'location_lon', 'uuid', 'score', 'rerank_score', 'address']]
    map_fig = filter_map(location)

    #drop columns that I dont want to show
    meta = df.drop(columns=["link", "node", "location_lat", "location_lon"])

    # Return the images, DataFrame, and map
    return images, meta, map_fig

def search(query):
    '''
    Send text query to a weaviate query and return results.

    NOTE: This will be similar to what we will have in our Sage data API.
    '''
    # send the query to Weaviate and get the results
    df = wq.clip_hybrid_query(query)

    #drop columns that I dont want
    results = df.drop(columns=["uuid"])
    cols = results.columns.tolist()

    # authorize results based on allowed nodes
    # TODO: implement auth using username and key from sage user
    results = results[results['vsn'].apply(lambda x: sq.authorize(x))]

    logging.debug("============FINAL RESULTS==================")
    logging.debug("auth filtering was completed.")
    logging.debug(results)
    logging.debug("===================END=====================")

    # if empty, tell Gradio exactly what to return (gradio changes the response when df is empty)
    if results.empty:
        results = {"headers": cols, "data": [], "metadata": None}
        return results

    # Return the results
    return results

# Gradio Interface Setup
def load_interface():
    '''
    Configure Gradio interface
    '''
    #set blocks
    iface_text_description = gr.Blocks()
    iface_upload_image = gr.Blocks()

    # text query tab
    with iface_text_description:

        # set title and description
        gr.Markdown(
        """
        # Text Query
        Enter a text description to find similar images.
        """)

        #set inputs
        query = gr.Textbox(label="Text Query", interactive=True)

        #Give examples
        queries=[["Show me images in Hawaii"], 
                 ["Rainy Chicago"], 
                 ["Snowy Mountains"], 
                 ["Show me clouds in the top camera"],
                 ["Cars in W049"],
                 ["W040"],
                 ["intersection in the right camera"]]
        examples = gr.Dataset(label="Example Queries", components=[query], samples=queries)

        #set buttons
        with gr.Row():
            sub_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")
            # Hidden button for search function
            hidden_search_btn = gr.Button("Hidden Search", visible=False, elem_classes=["hidden-button"])

        #set Outputs
        gr.Markdown(
        """
        Images Returned
        """)
        # col_widths = [
        #     "350px", #uuid
        #     "120px", #filename
        #     "1100px", #caption
        #     "180px", #score
        #     "1100px", #explain score
        #     "180px", #rerank score
        #     "90px", #vsn
        #     "150px", #camera
        #     "110px", #project
        #     "160px", #timestamp
        #     "250px", #host
        #     "250px", #job
        #     "500px", #plugin
        #     "220px", #task
        #     "100px", #zone
        #     "450px", #address
        # ]
        gallery = gr.Gallery( label="Returned Images", columns=[3], object_fit="contain", height="auto")
        meta = gr.DataFrame(label="Metadata", show_fullscreen_button=True, show_copy_button=True,) #column_widths=col_widths)
        plot = gr.Plot(label="Image Locations")
        hidden_results = gr.DataFrame(visible=False)

        #clear function
        def clear():
            return "", [], gr.DataFrame(value=None), gr.Plot(value=None)
        
        #select example func
        def on_select(evt: gr.SelectData):
            return evt.value[0]

        #set event listeners
        sub_btn.click(fn=text_query, inputs=query, outputs=[gallery, meta, plot])
        clear_btn.click(fn=clear, outputs=[query, gallery, meta, plot])  # Clear all components
        examples.select(fn=on_select, outputs=query)
        hidden_search_btn.click(fn=search, inputs=query, outputs=hidden_results)
        gr.SelectData

    # text Image query tab
    # with iface_upload_image: #TODO: Implement image_query() first

    #     # set title and description
    #     gr.Markdown(
    #     """
    #     # Image Query
    #     Upload an image to find similar images.
    #     """)

    #     #set inputs
    #     query = gr.File(label="Upload Image", file_types=['png', 'jpg', 'jpeg', 'gif','jfif'])

    #     #set buttons
    #     with gr.Row():
    #         sub_btn = gr.Button("Submit")
    #         clear_btn = gr.Button("Clear")

    #     #set Outputs
    #     certainty = gr.Textbox(label="Certainty Scores")
    #     gr.Markdown(
    #     """
    #     Images Returned
    #     """)
    #     gallery = gr.Gallery( label="Returned Images", columns=[3], object_fit="contain", height="auto")

    #     #clear function
    #     def clear():
    #         return None

    #     #set event listeners
    #     sub_btn.click(fn=image_query, inputs=query, outputs=[gallery, certainty])
    #     clear_btn.click(fn=clear, outputs=query)

    iface = gr.TabbedInterface(
        [iface_text_description],
        ["Text Query"]
        # [iface_text_description, iface_upload_image], Implement image_query() first
        # ["Text Query", "Image Query"] #TODO
    )
    
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)

def main():

    #start gradio app
    load_interface()

if __name__ == "__main__":
    main()
