# Model and training-data license review

**Checked:** 2026-10-24. This is a source review, not a legal opinion. The checkpoint model card identifies TESS, CREMA-D, SAVEE, and RAVDESS as training sources. These are listed separately because each has its own terms. The legal status of the checkpoint's derivative weights remains unverified pending review.

| Item | Stated license / access terms | Redistribution and commercial-use status | Verification and open questions |
| --- | --- | --- | --- |
| Fine-tuned checkpoint: `Dpngtm/wav2vec2-emotion-recognition` at `033a5751a5bbe5b0b67c2c71e6102c38de35a346` | Hugging Face repository metadata says MIT. | The MIT license generally permits use, modification, redistribution, and commercial use when its conditions are met. | Repository metadata was checked at the pinned snapshot and current model page. The dataset permissions below create unresolved questions about training and derivative weights. Do not treat the MIT tag as a definitive resolution of those questions. |
| Base model: `facebook/wav2vec2-base-960h` | Hugging Face repository metadata says Apache-2.0. | Apache-2.0 permits commercial use and redistribution subject to its license conditions and notices. | Repository metadata checked 2026-10-24. This describes the base repository's stated license; it does not decide the fine-tuned checkpoint's license chain. |
| Toronto Emotional Speech Set (TESS) | Borealis/University of Toronto Dataverse lists CC BY-NC 4.0. | The stated license excludes commercial use; attribution and other license terms apply. | Official dataset record checked 2026-10-24. The checkpoint card names this dataset but does not document the specific version, files, or permissions relied on during training. |
| CREMA-D | Official repository `LICENSE.txt` states Open Database License (ODbL) 1.0 for the database and Database Contents License (DbCL) 1.0 for individual contents. | ODbL permits commercial database use subject to its obligations; individual-content rights and attribution/redistribution conditions require separate review. | Official repository and license file checked 2026-10-24. The checkpoint card does not identify the exact CREMA-D materials or explain how database and individual-content terms were handled in training. |
| Surrey Audio-Visual Expressed Emotion (SAVEE) | Official registration page presents a database license agreement. It allows copies and derivatives for use within the licensee's organization and prohibits distribution or sublicensing of the database to third parties. | The published terms do not establish unrestricted public redistribution. Commercial-use permission is not established by the reviewed page. | Official registration/license terms checked 2026-10-24. Account/organization scope and the checkpoint training path need clarification from the dataset licensors. |
| RAVDESS | Official Zenodo record states CC BY-NC-SA 4.0 and says commercial licenses can be purchased. | The public license is noncommercial and share-alike. The record points users seeking commercial rights to a separate license. | Zenodo record checked 2026-10-24. The checkpoint card does not specify the version, files, or separate permission used. |

## Sources

- Checkpoint model card and pinned files: <https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition/tree/033a5751a5bbe5b0b67c2c71e6102c38de35a346>
- Checkpoint's current repository page: <https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition>
- Base model repository: <https://huggingface.co/facebook/wav2vec2-base-960h>
- TESS official Borealis record: <https://doi.org/10.5683/SP2/E8H2MF>
- CREMA-D official repository: <https://github.com/CheyneyComputerScience/CREMA-D>
- CREMA-D license file: <https://github.com/CheyneyComputerScience/CREMA-D/blob/master/LICENSE.txt>
- SAVEE official registration and license terms: <https://personalpages.surrey.ac.uk/p.jackson/SAVEE/Register.html>
- RAVDESS official Zenodo record: <https://doi.org/10.5281/zenodo.1188976>

## Pending review

The upstream dataset terms do not automatically establish that they transfer to a trained model, and no legal conclusion about derivative weights is made here. The checkpoint's MIT repository label also does not independently prove that all training-data permissions were sufficient for this use. Before public release or any commercial use, the team should confirm the checkpoint provenance and permissions with the checkpoint author, relevant dataset rights holders, and hackathon organizers. Do not commit or publicly redistribute any source dataset or demo recording without documented permission.
