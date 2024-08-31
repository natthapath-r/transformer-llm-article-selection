import json
import argparse
from tqdm import tqdm
from Bio import Entrez


def create_pubmed_article_json(file_path, output_name):
    # Get the PMID for a text file (separator is a line)
    with open(file_path, "r") as f:
        text = f.read()

    PMID_list = text.split("\n")

    # Split the list into chunks of 100
    chunk_size = 100
    PMID_chunks = [PMID_list[i : i + chunk_size] for i in range(0, len(PMID_list), chunk_size)]
    output_json = {}

    for chunk in tqdm(PMID_chunks):
        PMID_text = ",".join(chunk)

        handle = Entrez.efetch(db="pubmed", id=PMID_text, rettype="xml", retmode="text")
        records = Entrez.read(handle)

        # Target keys: PMID, DateCompleted (Day/Month/Year), ArticleTitle, Abstract, PublicationTypeList, MeshHeadingList
        for record in records["PubmedArticle"]:
            record_dict = {
                "DateCompleted": "",
                "ArticleTitle": "",
                "Abstract": "",
                "PublicationTypeList": [],
                "MeshHeadingList": [],
            }
            # Get the ID
            pmid = record["MedlineCitation"]["PMID"]

            # Add the date
            try:
                completed_date = []
                if "DateCompleted" not in record["MedlineCitation"]:
                    if record["MedlineCitation"]["Article"]["ArticleDate"] != []:
                        completed_date_dict = record["MedlineCitation"]["Article"]["ArticleDate"][-1]
                        if len(record["MedlineCitation"]["Article"]["ArticleDate"]) != 1:
                            print(f"Multiple dates found for PMID: {pmid}")
                    else:
                        completed_date_dict = record["MedlineCitation"]["DateRevised"]
                else:
                    completed_date_dict = record["MedlineCitation"]["DateCompleted"]

                if "Day" in completed_date_dict:
                    for key in ["Day", "Month", "Year"]:
                        completed_date.append(completed_date_dict[key])
                elif "Month" not in completed_date_dict:
                    completed_date.append(completed_date_dict["Year"])
                else:
                    for key in ["Month", "Year"]:
                        completed_date.append(completed_date_dict[key])
                completed_date = "/".join(completed_date)
            except KeyError:
                print(f"Date not found for PMID: {pmid}")

            # Add the title
            title = record["MedlineCitation"]["Article"]["ArticleTitle"]

            # Add the abstract
            abs = ""
            if "Abstract" in record["MedlineCitation"]["Article"]:
                for text_obj in record["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]:
                    abs += text_obj + " "
            abs = abs.strip()

            # Add the publication types
            pub_types = []
            for pub_type in record["MedlineCitation"]["Article"]["PublicationTypeList"]:
                pub_types.append(pub_type)

            # Add MeSH tags
            mesh_tags = []
            if "MeshHeadingList" in record["MedlineCitation"]:
                for mesh_obj in record["MedlineCitation"]["MeshHeadingList"]:
                    mesh_tags.append(mesh_obj["DescriptorName"])

            record_dict["DateCompleted"] = completed_date
            record_dict["ArticleTitle"] = title
            record_dict["Abstract"] = abs
            record_dict["PublicationTypeList"] = pub_types
            record_dict["MeshHeadingList"] = mesh_tags

            output_json[pmid] = record_dict

        for record in records["PubmedBookArticle"]:
            record_dict = {
                "DateCompleted": "",
                "ArticleTitle": "",
                "Abstract": "",
                "PublicationTypeList": [],
                "MeshHeadingList": [],
            }
            # Get the ID
            pmid = record["BookDocument"]["PMID"]
            # Get the date
            completed_date = []
            completed_date_dict = record["BookDocument"]["Book"]["PubDate"]
            for key in ["Month", "Year"]:
                completed_date.append(completed_date_dict[key])
            completed_date = "/".join(completed_date)
            # Get the title
            title = record["BookDocument"]["Book"]["BookTitle"]
            # Add the abstract
            abs = ""
            if "Abstract" in record["BookDocument"]:
                for text_obj in record["BookDocument"]["Abstract"]["AbstractText"]:
                    abs += text_obj + " "
            abs = abs.strip()
            # Add the publication types
            pub_types = []
            for pub_type in record["BookDocument"]["PublicationType"]:
                pub_types.append(pub_type)

            record_dict["DateCompleted"] = completed_date
            record_dict["ArticleTitle"] = title
            record_dict["Abstract"] = abs
            record_dict["PublicationTypeList"] = pub_types

            output_json[pmid] = record_dict

        handle.close()

    print("Number of articles: ", len(output_json))

    # Export to JSON
    with open(f"{output_name}.json", "w") as f:
        json.dump(output_json, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to the file containing the PMIDs")
    parser.add_argument("--output_name", type=str, help="Name of the output JSON file")
    args = parser.parse_args()

    create_pubmed_article_json(args.file_path, args.output_name)
