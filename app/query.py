'''This file implements functions that fetch results from weaviate for the query 
entered by user.'''
#NOTE: This will be deployed in our cloud under k8s namespace beehive-sage 
#   most likely integrated with beehive-data-api. If we want to allow our 
#   python client to use weaviate queries, we will also have to update our
#   sage-data-client python lib to include these new queries.

import HyperParameters as hp
from weaviate.classes.query import MetadataQuery, Move, HybridVector, Rerank, HybridFusion
from model import get_colbert_embedding, get_allign_embeddings, get_clip_embeddings
import logging
import requests
import os
from PIL import Image
from io import BytesIO
import pandas as pd

class Weav_query:
    """
    This class is used to query Weaviate.
    It contains methods for multiple types of queries.
    """

    def __init__(self, weav_client, triton_client=None):
        self.weav_client = weav_client
        self.triton_client = triton_client

    def hybrid_query(self, nearText, collection_name="HybridSearchExample"):
        """
        This method performs a hybrid vector and keyword search on a embedding space.
        """
        # used this for hybrid search params https://weaviate.io/developers/weaviate/search/hybrid

        #get collection
        collection = self.weav_client.collections.get(collection_name)

        # Perform the hybrid search
        res = collection.query.hybrid(
            query=nearText,  # The model provider integration will automatically vectorize the query
            target_vector="imagebind",  # The name of the vector space to search in
            fusion_type= HybridFusion.RELATIVE_SCORE,
            # max_vector_distance=hp.max_vector_distance,
            auto_limit=hp.autocut_jumps,
            limit=hp.response_limit,
            alpha=hp.query_alpha,
            return_metadata=MetadataQuery(score=True, explain_score=True),
            query_properties=["caption", "camera", "host", "job", "vsn", "plugin", "zone", "project", "address"], #Keyword search properties
            # bm25_operator=hp.keyword_search_params,
            vector=HybridVector.near_text(
                query=nearText,
                move_away=Move(force=hp.avoid_concepts_force, concepts=hp.concepts_to_avoid), #can this be used as guardrails?
                # distance=hp.max_vector_distance,
                # certainty=hp.near_text_certainty,
            ),
            rerank=Rerank(
                prop="caption", # The property to rerank on
                query=nearText  # If not provided, the original query will be used
            )
        )

        # init
        objects = []

        # Log the results
        logging.debug("============hybrid_query RESULTS==================")

        # Extract results from QueryReturn object type
        for obj in res.objects:
            #log results
            logging.debug("----------------%s----------------", obj.uuid)
            logging.debug(f"Properties: {obj.properties}")
            logging.debug(f"Score: {obj.metadata.score}")
            logging.debug(f"Explain Score: {obj.metadata.explain_score}")
            logging.debug(f"Rerank Score: {obj.metadata.rerank_score}")

            # Append the relevant object data into the list
            objects.append({
                "uuid": str(obj.uuid),
                "filename": obj.properties.get("filename", ""),
                "caption": obj.properties.get("caption", ""),
                "score": obj.metadata.score,
                "explainScore": obj.metadata.explain_score,
                "rerank_score": obj.metadata.rerank_score,
                "vsn": obj.properties.get("vsn", ""),
                "camera": obj.properties.get("camera", ""),
                "project": obj.properties.get("project", ""),
                "timestamp": obj.properties.get("timestamp", ""),
                "link": obj.properties.get("link", ""),
                "host": obj.properties.get("host", ""),
                "job": obj.properties.get("job", ""),
                "plugin": obj.properties.get("plugin", ""),
                "task": obj.properties.get("task", ""),
                "zone": obj.properties.get("zone", ""),
                "node": obj.properties.get("node", ""),
                "address": obj.properties.get("address", ""),
                "location_lat": self.get_location_coordinate(obj, "latitude"),
                "location_lon": self.get_location_coordinate(obj, "longitude"),
            })

        logging.debug("==============END========================")

        # Convert the list of dictionaries into a pandas DataFrame
        df = pd.DataFrame(objects)

        # Return the DataFrame
        return df

    def colbert_query(self, nearText, collection_name="HybridSearchExample"):
        """
        This method performs a vector search on a ColBERT embedding space.
        """
        #get collection
        collection = self.weav_client.collections.get(collection_name)

        # Generate colbert embedding
        colbert_embedding = get_colbert_embedding(self.triton_client, nearText)

        # Perform vector search on the "colbert" vector space
        res = collection.query.near_vector(
            near_vector=colbert_embedding,
            target_vector="colbert",
            auto_limit=hp.autocut_jumps,
            limit=hp.response_limit,
            # distance=hp.max_vector_distance,
            return_metadata=MetadataQuery(distance=True),
            rerank=Rerank(
                prop="caption", # The property to rerank on
                query=nearText  # If not provided, the original query will be used
            )
        )

        # init
        objects = []

        # Log the results
        logging.debug("============colberty_query RESULTS===============")

        # Extract results from QueryReturn object type
        for obj in res.objects:
            #log results
            logging.debug("----------------%s----------------", obj.uuid)
            logging.debug(f"Properties: {obj.properties}")
            logging.debug(f"Distance: {obj.metadata.distance}")
            logging.debug(f"Rerank Score: {obj.metadata.rerank_score}")

            # Append the relevant object data into the list
            objects.append({
                "uuid": str(obj.uuid),
                "filename": obj.properties.get("filename", ""),
                "caption": obj.properties.get("caption", ""),
                "distance": obj.metadata.distance,
                "rerank_score": obj.metadata.rerank_score,
                "vsn": obj.properties.get("vsn", ""),
                "camera": obj.properties.get("camera", ""),
                "project": obj.properties.get("project", ""),
                "timestamp": obj.properties.get("timestamp", ""),
                "link": obj.properties.get("link", ""),
                "host": obj.properties.get("host", ""),
                "job": obj.properties.get("job", ""),
                "plugin": obj.properties.get("plugin", ""),
                "task": obj.properties.get("task", ""),
                "zone": obj.properties.get("zone", ""),
                "node": obj.properties.get("node", ""),
                "address": obj.properties.get("address", ""),
                "location_lat": self.get_location_coordinate(obj, "latitude"),
                "location_lon": self.get_location_coordinate(obj, "longitude"),
            })

        logging.debug("==============END========================")

        # Convert the list of dictionaries into a pandas DataFrame
        df = pd.DataFrame(objects)

        # Return the DataFrame
        return df
    
    def colbert_hybrid_query(self, nearText, collection_name="HybridSearchExample"):
        """
        This method performs both a hybrid query on and a colbert query on either the same or seperate embedding spaces.
        Combines the results by normalizing their scores to [0, 1],
        then summing them according to specified weights. Deduplicates by UUID,
        and returns top results sorted by rerank_score. Weights must sum to 1.0.
        Final unified_score is between 0 and 1, where 0 is least relevant and 1 is most relevant.

        Parameters:
            nearText (str): The text query to search for.
            hybrid_collection (str): Name of the collection for hybrid search.
            vector_collection (str): Name of the collection for vector search.

        Returns:
            pd.DataFrame: Top-k deduplicated and reranked results.
        """
        # Ensure both weights add up to 1.0
        if hp.hybrid_weight + hp.colbert_weight != 1.0:
            raise ValueError("Weights must sum to 1.0")
        
        # Perform queries
        hybrid_df = self.hybrid_query(nearText, collection_name=collection_name)
        colbert_df = self.colbert_query(nearText, collection_name=collection_name)

        #NOTE: hybrid score is already normalized to [0, 1] by Weaviate,
        # Normalize vector 'distance'
        if not colbert_df.empty:
            min_score = colbert_df["distance"].min()
            max_score = colbert_df["distance"].max()
            colbert_df["normalized_vector_distance"] = (colbert_df["distance"] - min_score) / (max_score - min_score + 1e-10)
        else:
            colbert_df["normalized_vector_distance"] = []

        # Merge by uuid (outer join to keep all results)
        colbert_suffix = "_colbert"
        merged_df = pd.merge(
            hybrid_df,
            colbert_df,
            on="uuid",
            how="outer",
            suffixes=("", colbert_suffix)
        )

        # Dynamically identify and merge all shared columns
        for col in hybrid_df.columns:
            if col == "uuid":
                continue
            col_colbert = f"{col}{colbert_suffix}"
            if col_colbert in merged_df.columns:
                # Use hybrid value if present, fallback to colbert value
                merged_df[col] = merged_df[col].combine_first(merged_df[col_colbert])
                # Drop the duplicate colbert column
                merged_df.drop(columns=[col_colbert], inplace=True)

        # Fill missing scores if needed
        merged_df["normalized_vector_distance"] = merged_df["normalized_vector_distance"].fillna(0)
        merged_df["rerank_score"] = merged_df["rerank_score"].fillna(0)
        merged_df["score"] = merged_df["score"].fillna(0)

        merged_df["unified_score"] = (
            hp.hybrid_weight * merged_df["score"] +
            hp.colbert_weight * merged_df["normalized_vector_distance"]
        )

        # Sort and select top-k
        final_df = merged_df.sort_values(by=["rerank_score", "unified_score"], ascending=False).head(hp.hybrid_colbert_blend_top_k).reset_index(drop=True)

        # Logging block
        logging.debug("============colbert_hybrid_query RESULTS===============")
        for _,row in final_df.iterrows():
            logging.debug("----------------%s----------------", row["uuid"])
            logging.debug(f"Properties: {row.to_dict()}")
            logging.debug(f"Unified Score: {row.get('unified_score', 0):.4f}")
            # logging.debug(f"Normalized Hybrid Score: {row.get('normalized_hybrid_score', 0):.4f}")
            logging.debug(f"Hybrid Score: {row.get('score', 0):.4f}")
            # logging.debug(f"Normalized Vector Certainty: {row.get('normalized_vector_certainty', 0):.4f}")
            logging.debug(f"Normalized Vector Distance: {row.get('normalized_vector_distance', 0):.4f}")
            logging.debug(f"Rerank Score: {row.get('rerank_score', 0):.4f}")
        logging.debug("==============END========================")

        return final_df

    def get_location_coordinate(self, obj, coordinate_type):
        """ Helper function to safely fetch latitude or longitude from the location property. """
        location = obj.properties.get("location", "")
        if location:
            try:
                # Ensure the coordinate_type is valid and fetch the correct value
                return float(getattr(location, coordinate_type, "0.0")) if coordinate_type in ["latitude", "longitude"] else "0.0"
            except (AttributeError, ValueError):
                logging.warning(f"Invalid {coordinate_type} value found for obj {obj.uuid}")
                return "0.0"  # Default fallback for invalid location
        return "0.0"  # Default fallback if location is missing
    
    def clip_hybrid_query(self, nearText, collection_name="HybridSearchExample"):
        """
        This method performs a hybrid vector and keyword search on a clip embedding space.
        """
        # used this for hybrid search params https://weaviate.io/developers/weaviate/search/hybrid

        #get collection
        collection = self.weav_client.collections.get(collection_name)

        # get clip embedding
        clip_embedding = get_clip_embeddings(self.triton_client, nearText)

        # Perform the hybrid search
        res = collection.query.hybrid(
            query=nearText,  # The model provider integration will automatically vectorize the query
            target_vector="clip",  # The name of the vector space to search in
            fusion_type= HybridFusion.RELATIVE_SCORE,
            # max_vector_distance=hp.max_vector_distance,
            auto_limit=hp.autocut_jumps,
            limit=hp.response_limit,
            alpha=hp.query_alpha,
            return_metadata=MetadataQuery(score=True, explain_score=True),
            query_properties=["caption", "camera", "host", "job", "vsn", "plugin", "zone", "project", "address"], #Keyword search properties
            # bm25_operator=hp.keyword_search_params,
            vector=clip_embedding, # the custom vector
            # vector=HybridVector.near_text(
            #     query=nearText,
            #     move_away=Move(force=hp.avoid_concepts_force, concepts=hp.concepts_to_avoid), #can this be used as guardrails?
            #     # distance=hp.max_vector_distance,
            #     # certainty=hp.near_text_certainty,
            # ),
            rerank=Rerank(
                prop="caption", # The property to rerank on
                query=nearText  # If not provided, the original query will be used
            )
        )

        # init
        objects = []

        # Log the results
        logging.debug("============clip_hybrid_query RESULTS==================")

        # Extract results from QueryReturn object type
        for obj in res.objects:
            #log results
            logging.debug("----------------%s----------------", obj.uuid)
            logging.debug(f"Properties: {obj.properties}")
            logging.debug(f"Score: {obj.metadata.score}")
            logging.debug(f"Explain Score: {obj.metadata.explain_score}")
            logging.debug(f"Rerank Score: {obj.metadata.rerank_score}")

            # Append the relevant object data into the list
            objects.append({
                "uuid": str(obj.uuid),
                "filename": obj.properties.get("filename", ""),
                "caption": obj.properties.get("caption", ""),
                "score": obj.metadata.score,
                "explainScore": obj.metadata.explain_score,
                "rerank_score": obj.metadata.rerank_score,
                "vsn": obj.properties.get("vsn", ""),
                "camera": obj.properties.get("camera", ""),
                "project": obj.properties.get("project", ""),
                "timestamp": obj.properties.get("timestamp", ""),
                "link": obj.properties.get("link", ""),
                "host": obj.properties.get("host", ""),
                "job": obj.properties.get("job", ""),
                "plugin": obj.properties.get("plugin", ""),
                "task": obj.properties.get("task", ""),
                "zone": obj.properties.get("zone", ""),
                "node": obj.properties.get("node", ""),
                "address": obj.properties.get("address", ""),
                "location_lat": self.get_location_coordinate(obj, "latitude"),
                "location_lon": self.get_location_coordinate(obj, "longitude"),
            })

        logging.debug("==============END========================")

        # Convert the list of dictionaries into a pandas DataFrame
        df = pd.DataFrame(objects)

        # Return the DataFrame
        return df
    
class Sage_query:
    """
    This class is used to query Sage.
    """
    def __init__(self):
        return
    
    @staticmethod
    def _parse_deny_list(raw: str) -> set[str]:
        return {x.strip().lower() for x in raw.split(",") if x.strip()}

    def authorize(self, vsn: str, username: str = None, key: str = None) -> bool:
        """
        Default-allow. Deny only if VSN appears in UNALLOWED_NODES (case-insensitive).
        Returns False if vsn is empty/None.
        TODO: implement authorization logic using username and key from sage user
        """
        # in the meantime, we will use a static deny list from env var
        if not vsn:
            return False
        return vsn.strip().lower() not in self._parse_deny_list(os.getenv("UNALLOWED_NODES", ""))

    def getImage(self, url):
        '''
        Retrieve the Images from Sage
        '''
        #Creds
        USER = os.environ.get("SAGE_USER")
        PASS = os.environ.get("SAGE_PASS")

        # Auth header for Sage
        auth = (USER, PASS)

        try:
            # Get the image data
            response = requests.get(url, auth=auth)
            response.raise_for_status()  # Raise error for bad responses
            image_data = response.content

            # Convert the image data to a PIL Image
            image = Image.open(BytesIO(image_data))
            image = image.convert("RGB")  # Ensure it's in RGB mode if necessary

        except requests.exceptions.HTTPError as e:
            logging.debug(f"Image skipped, HTTPError for URL {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logging.debug(f"Image skipped, request failed for URL {url}: {e}")
            return None
        except Exception as e:
            logging.debug(f"Image skipped, an error occurred for URL {url}: {e}")
            return None

        return image

    # TODO: how will this be implemented in a hybrid search approach? create a caption for the image entered by the user?
    # def testImage(nearImage,client):
    #     # I am fetching top 3 results for the user, we can change this by making small 
    #     # altercations in this function and in upload.html file
    #     # # adding a threshold: https://weaviate.io/developers/weaviate/search/similarity#set-a-similarity-threshold
    #     imres = client.query.get("ClipExample", ["text", "_additional {certainty} "]).with_near_image(nearImage).do()
    #     return {
    #         "objects": ((imres['data']['Get']['ClipExample'][0]['text']),(imres['data']['Get']['ClipExample'][1]['text']),(imres['data']['Get']['ClipExample'][2]['text'])),
    #         "scores": (imres['data']['Get']['ClipExample'][0]['_additional'],imres['data']['Get']['ClipExample'][1]['_additional'],imres['data']['Get']['ClipExample'][2]['_additional'])
    #     }