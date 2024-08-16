# UltraCascade

This is a native adaptation of the UltraPixel model that facilitates the generation of high resolution latents by Stable Cascade by mitigating the tendency toward doubling artifacts and mutations when generating at an largely untrained resolution, by using an image generated at a native resolution as a "guide" (a bit like a good tile controlnet). The original implementation somewhat obscured this fact, as it generated the guide and the high resolution stage C latent within the same node. Here, everything is separated into discrete stages to maximize the creative possibilities and control over fidelity. The basic pipeline is:

Stage C -> Stage UP -> Stage B 

(followed by VAE decode, which is stage A)

This repo contains the code for the models themselves, and implements support for "Self-Attention Guidance" (SAG) in stages C and B. (Support for this in stage B requires replacing the stage_b.py file in your comfy folder with the one available here. The path is comfy/ldm/cascade). It also implements "Random Attention Guidance" (RAG), which is particularly effective for photography styles when specific combinations of positive and negative scales are used. (I recommend +0.2 for stage C and -0.1 for stage UP as a starting point, using DPMPP_SDE_ADVANCED with perlin noise, available in the RES4LYF node pack: https://github.com/ClownsharkBatwing/RES4LYF)

There are UltraCascade KSampler and KSamplerAdvanced nodes included in this repo. It is no longer necessary to use the Cascade Stage B Conditioning nodes, or the ConditioningZeroOut nodes, as the code is included within the samplers themselves. Simply link the output for stage C into the "guide" input of a subsequent sampler, and it'll detect which model type you've hooked up - if it's from the UltraCascade Loader, it'll use stage UP, and if it's a stage B model, it'll use stage B. 

The UltraSharkSampler node, available in RES4LYF, operates under similar principles but has much more advanced options. I highly recommend using this over the simpler alternatives. 

The UltraSharkSampler Tiled node is an option for adding additional detail and polish to an image to stage B. It borrows much of the tiling code from Blender Neko's excellent Tiled KSampler (https://github.com/BlenderNeko/ComfyUI_TiledKSampler), and is available in RES4LYF. It collects tiles into non-intersecting batches which results in a dramatic improvement in inference speed (2.5x+ on a 4090).

Cascade stages C and UP are particularly well behaved with perlin noise - not just as the initial noise, but used throughout the diffusion process. These options are available in the DPMPP_SDE_ADVANCED and ClownSampler nodes in RES4LYF. (ClownSampler has a huge number of unique features, and uses the RES sampler as the backend, with the core sampling code adapted from https://github.com/Clybius/ComfyUI-Extra-Samplers/). The two samplers are very complementary with Cascade. DPMPP_SDE affords a more "organic" look, but is somewhat more prone to artifacts, while ClownSampler is unrivaled at refined, "clean" images. 

Cascade stage B has by far the lowest artifact rate when using "pyramid-cascade_B" noise with ClownSampler. This is a custom stochastic noise mode that emulates the same noise it was trained with. It's an unusual strategy, but it's very effective at putting "nasty Cascade noise" out to pasture. This noise mode is only available in the RES4LYF repo.

Have you noticed I've mentioned RES4LYF a lot? I have. That's because it's the greatest repo to ever repo.

