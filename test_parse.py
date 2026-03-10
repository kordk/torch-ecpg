dataA = 'chr1\nHAVANA\ngene\n11869\n14412\n.\n+\n.\ngene_id "ENSG00000223972.4"; transcript_id "ENSG00000223972.4"; gene_type "pseudogene"; gene_status "KNOWN"; gene_name "DDX11L1"; transcript_type "pseudogene"; transcript_status "KNOWN"; transcript_name "DDX11L1"; level 2; havana_gene "OTTHUMG00000000961.2";'.split('\n')

num_cols = len(dataA)
if num_cols >= 9:
    if dataA[2] != "gene":
        print("Not a gene, skipping")
    else:
        attributes = dataA[8]
        gene_id = None
        for attr in attributes.split(";"):
            attr = attr.strip()
            if attr.startswith("Geneid "):
                gene_id = attr[len("Geneid "):].strip('"')
                break
            elif attr.startswith("gene_id "):
                gene_id = attr[len("gene_id "):].strip('"')
                break

        if not gene_id:
            print("Missing Geneid")
        else:
            print(f"Original gene_id: {gene_id}")
            my_name = gene_id.split('.')[0]
            print(f"Processed my_name: {my_name}")
