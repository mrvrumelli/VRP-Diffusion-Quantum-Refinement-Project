To: ruibin.bai@nottingham.edu.cn
Cc: zhenwei.wang@nottingham.edu.cn
Subject: Reproduction questions on "Constraints Matrix Diffusion based Generative Neural Solver for Vehicle Routing Problems" (arXiv:2603.07568)

Dear Professor Bai and co-authors,

I am reproducing the graph-diffusion constraint-matrix component of your paper as part of a
research project on CVRP solving, and I would be very grateful for your help with a few details.

Our implementation follows the paper as closely as we can: 50,000 HGS-labelled instances (N = 20,
50 and 100, uniform coordinates, random depot, capacities 30/40/50), a pretrained five-layer GAT,
the anisotropic gated denoiser of Eqs. 10–15, T = 1000 with a linear schedule from 1e-4 to 0.02,
50 epochs, and 50 inference steps. Our constraint-matrix F1 is about 0.53 at 50 steps, compared
with 0.823 in Figure 8, and our accuracy is 0.90 compared with 0.95 in Figure 6. Our F1 is also
nearly flat across step counts (0.55 at one step), whereas Figure 8 rises from 0.616 to 0.823. We
want to make sure we are measuring the same quantity before drawing conclusions.

The questions that matter most to us:

1. F1 definition (Figures 6 and 8). Is it the positive-class F1 or a macro average over both
   classes? Is it pooled over all matrix entries or averaged per instance? Are diagonal entries
   included? Which threshold, which problem sizes, and how many instances were used?
2. Validation set. Were the validation instances generated separately, or split from the
   training pool? If split, was this done before or after augmentation, so that augmented copies
   of training instances could not appear in validation?
3. Inference. In Section V-B.4, what does "we do not perform per-step sampling" mean: are
   intermediate states sampled from the posterior, or propagated deterministically? Which timestep
   subsequence and reverse transition are used for 50 steps, and is the final matrix the
   thresholded probability or the last sample?
4. GAT input and training. Does the GAT used by the diffusion model include the depot as a
   node, and which node features does it use? Is it frozen or fine-tuned while training the
   denoiser, and what objective was it pretrained with?
5. Training objective. Is the denoiser trained on the variational bound of Eq. 7, or with a
   cross-entropy on x0 as in DIFUSCO? Is any class weighting applied to the sparse positive class?

Secondary details, if convenient:

6. Does Q_t flip a bit with probability β_t as printed, or β_t / 2 as in the DIFUSCO code?
7. Is one model trained across all sizes, or one per size?
8. HGS version, time budget or stopping rule, seeds, and whether any labels were filtered.
9. How the geometric and demand augmentations are combined, and whether identity is included.
10. Does the denoiser's edge input contain x_t only, or also distances?

If you are able to share code, a checkpoint, or the evaluation script, even in part, that would
answer most of these at once. Thank you very much for your time and for the interesting work.

Best regards,
Mustafa Mert
[affiliation]
