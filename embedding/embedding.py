import numpy as np



#np.savez_compressed("atir_embeddings.npz", new_train_faces=new_train_faces, train_labels=train_labels)
#np.savez_compressed("mean_embeddings.npz", embed_michal=embed_michal, embed_milosz=embed_milosz)

with np.load("../atir_embeddings.npz") as data:
    train_faces, train_labels = data["new_train_faces"], data["train_labels"]
    
print(train_faces.shape)
print(train_faces[0])
print(train_faces[-1])
    
with np.load("../mean_embeddings.npz") as data:
    embed_michal, embed_milosz = data["embed_michal"], data["embed_milosz"]
    
print(embed_michal)
print(embed_milosz)

