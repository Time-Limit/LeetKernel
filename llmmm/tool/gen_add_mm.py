for block_tile_m in [16, 32, 64, 128]:
    for block_tile_n in [8, 16, 32, 64, 128, 256]:
        for thread_tile_m in [4, 8, 16, 32, 64, 128]:
            for thread_tile_n in [4, 8, 16, 32, 64, 128]:
                for tile_k in [8, 16, 32, 64, 128]:
                    for split_k in [1, 2, 4, 8, 16, 32]:
                        for reduce_block_tile in [1, 128, 256, 512, 1024]:
                            for reduce_thread_tile in [1, 4, 8, 16, 32]:
                                if block_tile_m % thread_tile_m != 0:
                                    print(f"{block_tile_m=:3d}, {block_tile_n=:3d}, {tile_k=:3d}, {thread_tile_m=:3d}, {thread_tile_n=:3d}, point 0");
                                    continue
                                if block_tile_n % thread_tile_n != 0:
                                    print(f"{block_tile_m=:3d}, {block_tile_n=:3d}, {tile_k=:3d}, {thread_tile_m=:3d}, {thread_tile_n=:3d}, point 1");
                                    continue
                                m_thread = block_tile_m // thread_tile_m
                                n_thread = block_tile_n // thread_tile_n
                                thread_count = m_thread * n_thread
                                if (thread_count & (thread_count-1)) != 0 or thread_count % 32 != 0:
                                    print(f"{block_tile_m=:3d}, {block_tile_n=:3d}, {tile_k=:3d}, {thread_tile_m=:3d}, {thread_tile_n=:3d}, point 2");
                                    continue
                                if (thread_count > 256):
                                    print(f"{block_tile_m=:3d}, {block_tile_n=:3d}, {tile_k=:3d}, {thread_tile_m=:3d}, {thread_tile_n=:3d}, point 3");
                                    continue
                                bytes_per_float = 4;
                                buffer_size = 4;
                                shared_memory_bytes = (block_tile_m + block_tile_n) * tile_k * bytes_per_float * buffer_size;
                                total_sm_bytes_per_sm_4090 = 100 * 1024
                                sm_bytes_runtime_used = 1024;
                                if shared_memory_bytes > total_sm_bytes_per_sm_4090 - sm_bytes_runtime_used:
                                    print(f"{block_tile_m=:3d}, {block_tile_n=:3d}, {tile_k=:3d}, {thread_tile_m=:3d}, {thread_tile_n=:3d}, point 4");
                                    continue
                                if block_tile_m * tile_k % thread_count != 0:
                                    print(f"{block_tile_m=:3d}, {block_tile_n=:3d}, {tile_k=:3d}, {thread_tile_m=:3d}, {thread_tile_n=:3d}, point 5");
                                    continue
                                if block_tile_n * tile_k % thread_count != 0:
                                    print(f"{block_tile_m=:3d}, {block_tile_n=:3d}, {tile_k=:3d}, {thread_tile_m=:3d}, {thread_tile_n=:3d}, point 6");
                                    continue
                                A_LDG_REG_COUNT = block_tile_m * tile_k // thread_count
                                B_LDG_REG_COUNT = block_tile_n * tile_k // thread_count
                                if A_LDG_REG_COUNT % 4 != 0:
                                    print(f"{block_tile_m=:3d}, {block_tile_n=:3d}, {tile_k=:3d}, {thread_tile_m=:3d}, {thread_tile_n=:3d}, point 7");
                                    continue
                                if B_LDG_REG_COUNT % 4 != 0:
                                    print(f"{block_tile_m=:3d}, {block_tile_n=:3d}, {tile_k=:3d}, {thread_tile_m=:3d}, {thread_tile_n=:3d}, point 8");
                                    continue
                                m_cal_thread = block_tile_m // thread_tile_m
                                n_cal_thread = block_tile_n // thread_tile_n
                                if m_cal_thread * n_cal_thread != thread_count:
                                    print(f"{block_tile_m=:3d}, {block_tile_n=:3d}, {tile_k=:3d}, {thread_tile_m=:3d}, {thread_tile_n=:3d}, point 9");
                                    continue
                                register_count_per_thread = (A_LDG_REG_COUNT + B_LDG_REG_COUNT) * 2 + thread_tile_m * 2 + thread_tile_n * 2 + thread_tile_m * thread_tile_n
                                if register_count_per_thread > 224:
                                    print(f"{block_tile_m=:3d}, {block_tile_n=:3d}, {tile_k=:3d}, {thread_tile_m=:3d}, {thread_tile_n=:3d}, point 10, {register_count_per_thread=}");
                                    continue
                                total_register_count = (register_count_per_thread * thread_count + 255-1) // 255 * 255
                                if total_register_count > 65535:
                                    print(f"{block_tile_m=:3d}, {block_tile_n=:3d}, {tile_k=:3d}, {thread_tile_m=:3d}, {thread_tile_n=:3d}, point 11");
                                    continue
                                if block_tile_m == 128 and split_k != 1:
                                    continue
                                if split_k == 1 and (reduce_block_tile != 1 or reduce_thread_tile != 1):
                                    continue
                                if split_k != 1 and (reduce_block_tile == 1 or reduce_thread_tile == 1):
                                    continue
                                if split_k != 1:
                                    if reduce_block_tile % reduce_thread_tile != 0:
                                        continue
                                    thread_count = reduce_block_tile // reduce_thread_tile
                                    if thread_count < 32 or thread_count > 256:
                                        continue
                                    if thread_count & (thread_count - 1):
                                        continue
                                    if reduce_thread_tile + reduce_thread_tile * 2 > 192: # register
                                        continue
                                print(f"if constexpr (BLOCK_TILE_M == {block_tile_m:4d}) {{ AddMM(BLOCK_TILE_M, {block_tile_n:4d}, {thread_tile_m:4d}, {thread_tile_n:4d}, {tile_k:4d}, {split_k:4d}, {reduce_block_tile:4d}, {reduce_thread_tile:4d}); }} // {block_tile_m=:4d}, {block_tile_n=:4d}, {tile_k=:4d}, {thread_tile_m=:4d}, {thread_tile_n=:4d}, {thread_count=:4d}, {register_count_per_thread=:4d}, {total_register_count=:5d}, {shared_memory_bytes=:6d}");
