# print("A priori")
#
# v_cardinal = np.zeros(n_pop)
# for i in range(n_pop):
#     for mu in range(p):
#         v_cardinal[i] += v_pop[i, mu]
#
# s_hat = np.zeros(n_pop)
# for i in range(n_pop):
#     s_hat[i] = (1-f) ** (p-v_cardinal[i]) * f**(v_cardinal[i])
#
# print("s hat", s_hat)
# print("s", s)
#
# fig, ax = plt.subplots()
#
# ax.scatter(np.arange(n_pop), s, color="C0", alpha=0.4)
# ax.scatter(np.arange(n_pop), s_hat, color="C1", alpha=0.4)
# plt.show()
#
# print(v_pop[0])
#
# print("v cardinal", v_cardinal)




# memory_patterns = []
# pbar = tqdm(total=n)
# while True:
#     pattern = \
#         np.random.choice([0, 1], p=[1 - f, f], size=p)
#     if not np.sum(pattern):
#         continue
#     memory_patterns.append(pattern)
#     pbar.update(1)
#     if len(memory_patterns) == n:
#         break
# pbar.close()
