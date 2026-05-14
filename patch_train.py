import re

with open("train.py", "r") as f:
    content = f.read()

# Replace Discriminator imports
content = content.replace("from models.discriminator import Discriminator", "from models.discriminator import get_D1, get_D2")

# Update dwt_loss
dwt_old = """def dwt_loss(x, y):
    \"\"\"FIX: Emphasize low-freq (visible) more, ignore high-freq mismatch.\"\"\"
    llx, lhx, hlx, hhx = haar_dwt2d(x)
    lly, lhy, hly, hhy = haar_dwt2d(y)
    return (F.l1_loss(llx, lly)
            + 0.1 * (F.l1_loss(lhx, lhy) + F.l1_loss(hlx, hly) + F.l1_loss(hhx, hhy)))"""

dwt_new = """def dwt_loss(x, y):
    \"\"\"Full Haar DWT Loss: MSE across high-frequency subbands.\"\"\"
    llx, lhx, hlx, hhx = haar_dwt2d(x)
    lly, lhy, hly, hhy = haar_dwt2d(y)
    # Task 3: MSE across all high-frequency subbands to mimic high-frequency noise
    return F.mse_loss(lhx, lhy) + F.mse_loss(hlx, hly) + F.mse_loss(hhx, hhy)"""
content = content.replace(dwt_old, dwt_new)

# Update phase defaults
content = content.replace("parser.add_argument('--epochs', type=int, default=200)", "parser.add_argument('--epochs', type=int, default=160)")
content = content.replace("parser.add_argument('--phase1_epoch', type=int, default=30)", "parser.add_argument('--phase1_epoch', type=int, default=15)")
content = content.replace("parser.add_argument('--phase2_epoch', type=int, default=80)", "parser.add_argument('--phase2_epoch', type=int, default=130)")
content = content.replace("parser.add_argument('--phase3_epoch', type=int, default=160)", "parser.add_argument('--phase3_epoch', type=int, default=160)")

# Update D instantiation
d_old = "    D = Discriminator().to(device)"
d_new = """    D1 = get_D1().to(device)
    D2 = get_D2().to(device)"""
content = content.replace(d_old, d_new)

d_para_old = """    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)"""
d_para_new = """    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        G = nn.DataParallel(G)
        D1 = nn.DataParallel(D1)
        D2 = nn.DataParallel(D2)"""
content = content.replace(d_para_old, d_para_new)

# Update optimizers
opt_d_old = "    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_early, betas=(0.5, 0.999))"
opt_d_new = "    opt_d = torch.optim.Adam(list(D1.parameters()) + list(D2.parameters()), lr=args.lr_early, betas=(0.5, 0.999))"
content = content.replace(opt_d_old, opt_d_new)

# Update scheduler step logic and mode logic
train_prep_old = """    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train()"""
train_prep_new = """    for epoch in range(start_epoch, args.epochs):
        G.train(); D1.train(); D2.train()"""
content = content.replace(train_prep_old, train_prep_new)

# Update loss computation inside train loop
# We need to change lambda_lpips to 0.3
cover_loss_old = "cover_loss  = L_cover_mse + 0.5 * L_cover_l1 + 0.3 * L_dwt + 0.1 * L_lpips"
cover_loss_new = "cover_loss  = L_cover_mse + 0.5 * L_cover_l1 + 0.3 * L_dwt + 0.3 * L_lpips"
content = content.replace(cover_loss_old, cover_loss_new)

# Discriminator and adversarial loss handling inside the loop
adv_code_old = """                if epoch >= args.phase2_epoch:
                    d_fake_g = D(watermarked.float())
                    # FIX: bumped adv weight from 0.01 → 0.05
                    g_loss = g_loss + 0.125 * generator_adv_loss(d_fake_g)"""

adv_code_new = """                if epoch >= args.phase1_epoch:  # Phase 2 starts at phase1_epoch
                    # Adversarial loss incorporating both D1 and D2
                    d1_fake_g = D1(watermarked.float())

                    # D2 expects watermarked image concatenated with its saliency map
                    mask_w = G.module.saliency(watermarked).detach() if isinstance(G, nn.DataParallel) else G.saliency(watermarked).detach()
                    d2_input = torch.cat([watermarked.float(), mask_w], dim=1)
                    d2_fake_g = D2(d2_input)

                    lambda_d = 1.0
                    l_adv = generator_adv_loss(d1_fake_g) + lambda_d * generator_adv_loss(d2_fake_g)
                    g_loss = g_loss + l_adv"""
content = content.replace(adv_code_old, adv_code_new)

# Discriminator update loop
d_update_old = """            d_loss_val = 0.0
            if epoch >= args.phase2_epoch:
                opt_d.zero_grad(set_to_none=True)
                real_img = cover.detach().float()
                fake_img = watermarked.detach().float()
                d_real = D(real_img)
                d_fake = D(fake_img)
                d_loss = discriminator_loss(d_real, d_fake)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                opt_d.step()
                d_loss_val = d_loss.item()"""

d_update_new = """            d_loss_val = 0.0
            if epoch >= args.phase1_epoch:
                opt_d.zero_grad(set_to_none=True)
                real_img = cover.detach().float()
                fake_img = watermarked.detach().float()

                # D1 Update
                d1_real = D1(real_img)
                d1_fake = D1(fake_img)
                d1_loss = discriminator_loss(d1_real, d1_fake)

                # D2 Update
                mask_real = G.module.saliency(real_img).detach() if isinstance(G, nn.DataParallel) else G.saliency(real_img).detach()
                mask_fake = G.module.saliency(fake_img).detach() if isinstance(G, nn.DataParallel) else G.saliency(fake_img).detach()

                d2_real_input = torch.cat([real_img, mask_real], dim=1)
                d2_fake_input = torch.cat([fake_img, mask_fake], dim=1)

                d2_real = D2(d2_real_input)
                d2_fake = D2(d2_fake_input)
                d2_loss = discriminator_loss(d2_real, d2_fake)

                d_loss = d1_loss + d2_loss
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(D1.parameters()) + list(D2.parameters()), 1.0)
                opt_d.step()
                d_loss_val = d_loss.item()"""

content = content.replace(d_update_old, d_update_new)

# In the curriculum setup logic:
curr_old = """        # FIX: Curriculum learning - no attacks in Phase 1
        if epoch < args.phase1_epoch:
            current_attack = identity_only
            lambda_secret = 2.0       # heavily emphasize secret first
            lambda_cover  = 1.0
        elif epoch < args.phase2_epoch:
            current_attack = attack_module
            lambda_secret = 1.5
            lambda_cover  = 1.0
        else:
            current_attack = attack_module
            lambda_secret = 1.0
            lambda_cover  = 1.5"""

curr_new = """        # Curriculum learning
        if epoch < args.phase1_epoch:
            # Phase 1: Bijective training, Identity noise, no adversarial
            current_attack = identity_only
            lambda_secret = 2.0
            lambda_cover  = 1.0
        elif epoch < args.phase2_epoch:
            # Phase 2: D1/D2 active, attack active
            current_attack = attack_module
            lambda_secret = 1.5
            lambda_cover  = 1.0
        else:
            # Phase 3: Fine tuning
            current_attack = attack_module
            lambda_secret = 1.0
            lambda_cover  = 1.5"""
content = content.replace(curr_old, curr_new)

# One more phase2_epoch -> phase1_epoch usage in scheduling? Actually it's phase2_epoch in the original, now Phase 2 starts at phase1_epoch so we should update this:
sch_d_step_old = """        if epoch >= args.phase2_epoch:
            sch_d.step()"""
sch_d_step_new = """        if epoch >= args.phase1_epoch:
            sch_d.step()"""
content = content.replace(sch_d_step_old, sch_d_step_new)

# Pre-loop sch_d step
pre_sch_d_step_old = """        if start_epoch >= args.phase2_epoch:
            sch_d.step()"""
pre_sch_d_step_new = """        if start_epoch >= args.phase1_epoch:
            sch_d.step()"""
content = content.replace(pre_sch_d_step_old, pre_sch_d_step_new)


with open("train.py", "w") as f:
    f.write(content)
