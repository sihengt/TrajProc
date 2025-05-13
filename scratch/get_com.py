# def body_total_com(body_uid):
#     """
#     Returns (com_world, total_mass) for a multibody, where com_world is an (x,y,z)
#     tuple in world coordinates.
#     """
#     n_links = p.getNumJoints(body_uid)
#     total_mass   = 0.0
#     com_sum      = np.zeros(3)

#     # ---------------- base (-1) and every link ----------------
#     for link in range(-1, n_links):
#         mass, _, _, local_inertial_pos, local_inertial_orn, *_ = \
#             p.getDynamicsInfo(body_uid, link)      # has mass & local CoM :contentReference[oaicite:0]{index=0}
#         if mass == 0:                 # static or visual‑only part
#             continue

#         if link == -1:                # base link
#             link_world_pos, link_world_orn = p.getBasePositionAndOrientation(body_uid)
#         else:
#             link_state                = p.getLinkState(body_uid, link, computeForwardKinematics=1)
#             link_world_pos, link_world_orn = link_state[0], link_state[1]

#         # transform the local inertial offset into world frame
#         com_i, _ = p.multiplyTransforms(link_world_pos, link_world_orn,
#                                          local_inertial_pos, local_inertial_orn)
#         com_sum      += mass * np.array(com_i)
#         total_mass   += mass

#     return tuple(com_sum / total_mass), total_mass

# def drop_debug_marker(world_pos, colour=[0,1,0], life=0):
#     sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=colour+[1])
#     p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere,
#                       basePosition=world_pos)