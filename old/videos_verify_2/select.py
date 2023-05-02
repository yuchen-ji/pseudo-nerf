with open("/home/shenxi/interns/JiYuchen/semi-nerf/videos_verify_2/verify_1v_10v.txt", 'r') as r:
    lines = r.readlines()
with open("/home/shenxi/interns/JiYuchen/semi-nerf/videos_verify_2/1.txt", "w") as w:
    for l in lines:
        if "test_psnr" in l or "Instance" in l:
            w.write(l)