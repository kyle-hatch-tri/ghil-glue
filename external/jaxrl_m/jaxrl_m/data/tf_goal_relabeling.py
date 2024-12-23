"""
Contains goal relabeling and reward logic written in TensorFlow.

Each relabeling function takes a trajectory with keys "observations",
"next_observations", and "terminals". It returns a new trajectory with the added
keys "goals", "rewards", and "masks". Keep in mind that "observations" and
"next_observations" may themselves be dictionaries, and "goals" must match their
structure.

"masks" determines when the next Q-value is masked out. Typically this is NOT(terminals).
"""

import tensorflow as tf


def uniform(traj, *, reached_proportion):
    """
    Relabels with a true uniform distribution over future states. With
    probability reached_proportion, observations[i] gets a goal
    equal to next_observations[i].  In this case, the reward is 0. Otherwise,
    observations[i] gets a goal sampled uniformly from the set
    next_observations[i + 1:], and the reward is -1.
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # select a random future index for each transition i in the range [i + 1, traj_len)
    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len) + 1, tf.float32)
    high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # TODO(kvablack): don't know how I got an out-of-bounds during training,
    # could not reproduce, trying to patch it for now
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    # select a random proportion of transitions to relabel with the next observation
    goal_reached_mask = tf.random.uniform([traj_len]) < reached_proportion

    # the last transition must be goal-reaching
    goal_reached_mask = tf.logical_or(
        goal_reached_mask, tf.range(traj_len) == traj_len - 1
    )

    # make goal-reaching transitions have an offset of 0
    goal_idxs = tf.where(goal_reached_mask, tf.range(traj_len), goal_idxs)

    # select goals
    traj["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        traj["next_observations"],
    )

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = tf.cast(tf.where(goal_reached_mask, 0, -1), tf.int32)

    # add masks
    traj["masks"] = tf.logical_not(traj["terminals"])

    return traj


def last_state_upweighted(traj, *, reached_proportion):
    """
    A weird relabeling scheme where the last state gets upweighted. For each
    transition i, a uniform random number is generated in the range [i + 1, i +
    traj_len). It then gets clipped to be less than traj_len. Therefore, the
    first transition (i = 0) gets a goal sampled uniformly from the future, but
    for i > 0 the last state gets more and more upweighted.
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # select a random future index for each transition
    offsets = tf.random.uniform(
        [traj_len],
        minval=1,
        maxval=traj_len,
        dtype=tf.int32,
    )

    # select random transitions to relabel as goal-reaching
    goal_reached_mask = tf.random.uniform([traj_len]) < reached_proportion
    # last transition is always goal-reaching
    goal_reached_mask = tf.logical_or(
        goal_reached_mask, tf.range(traj_len) == traj_len - 1
    )

    # the goal will come from the current transition if the goal was reached
    offsets = tf.where(goal_reached_mask, 0, offsets)

    # convert from relative to absolute indices
    indices = tf.range(traj_len) + offsets

    # clamp out of bounds indices to the last transition
    indices = tf.minimum(indices, traj_len - 1)

    # select goals
    traj["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, indices),
        traj["next_observations"],
    )

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = tf.cast(tf.where(goal_reached_mask, 0, -1), tf.int32)

    # add masks
    traj["masks"] = tf.logical_not(traj["terminals"])

    return traj


def geometric(traj, *, reached_proportion, discount):
    """
    Relabels with a geometric distribution over future states. With
    probability reached_proportion, observations[i] gets a goal
    equal to next_observations[i].  In this case, the reward is 0. Otherwise,
    observations[i] gets a goal sampled geometrically from the set
    next_observations[i + 1:], and the reward is -1.
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # geometrically select a future index for each transition i in the range [i + 1, traj_len)
    arange = tf.range(traj_len)
    is_future_mask = tf.cast(arange[:, None] < arange[None], tf.float32)
    d = discount ** tf.cast(arange[None] - arange[:, None], tf.float32)

    probs = is_future_mask * d
    # The indexing changes the shape from [seq_len, 1] to [seq_len]
    goal_idxs = tf.random.categorical(
        logits=tf.math.log(probs), num_samples=1, dtype=tf.int32
    )[:, 0]

    # select a random proportion of transitions to relabel with the next observation
    goal_reached_mask = tf.random.uniform([traj_len]) < reached_proportion

    # the last transition must be goal-reaching
    goal_reached_mask = tf.logical_or(
        goal_reached_mask, tf.range(traj_len) == traj_len - 1
    )

    # make goal-reaching transitions have an offset of 0
    goal_idxs = tf.where(goal_reached_mask, tf.range(traj_len), goal_idxs)

    # select goals
    traj["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        traj["next_observations"],
    )

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = tf.cast(tf.where(goal_reached_mask, 0, -1), tf.int32)

    # add masks
    traj["masks"] = tf.logical_not(traj["terminals"])

    return traj


def delta_goals(traj, *, goal_delta):
    """
    Relabels with a uniform distribution over future states in the range [i +
    goal_delta[0], min{traj_len, i + goal_delta[1]}). Truncates trajectories to
    have length traj_len - goal_delta[0]. Not suitable for RL (does not add
    terminals or rewards).
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # add the last observation (which only exists in next_observations) to get
    # all the observations
    all_obs = tf.nest.map_structure(
        lambda obs, next_obs: tf.concat([obs, next_obs[-1:]], axis=0),
        traj["observations"],
        traj["next_observations"],
    )
    all_obs_len = traj_len + 1

    # current obs should only come from [0, traj_len - goal_delta[0])
    curr_idxs = tf.range(traj_len - goal_delta[0])

    # select a random future index for each transition i in the range [i + goal_delta[0], min{all_obs_len, i + goal_delta[1]})
    rand = tf.random.uniform([traj_len - goal_delta[0]])
    low = tf.cast(curr_idxs + goal_delta[0], tf.float32)
    high = tf.cast(tf.minimum(all_obs_len, curr_idxs + goal_delta[1]), tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # very rarely, floating point errors can cause goal_idxs to be out of bounds
    goal_idxs = tf.minimum(goal_idxs, all_obs_len - 1)

    traj_truncated = tf.nest.map_structure(
        lambda x: tf.gather(x, curr_idxs),
        traj,
    )

    # select goals
    traj_truncated["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        all_obs,
    )

    traj_truncated["goal_dists"] = goal_idxs - curr_idxs

    return traj_truncated


def delta_goals2(traj, *, goal_delta):
    """
    Relabels with a uniform distribution over future states in the range [i +
    goal_delta[0], min{traj_len, i + goal_delta[1]}). Truncates trajectories to
    have length traj_len - goal_delta[0]. Not suitable for RL (does not add
    terminals or rewards).
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # add the last observation (which only exists in next_observations) to get
    # all the observations
    all_obs = tf.nest.map_structure(
        lambda obs, next_obs: tf.concat([obs, next_obs[-1:]], axis=0),
        traj["observations"],
        traj["next_observations"],
    )
    all_obs_len = traj_len + 1

    # current obs should only come from [0, traj_len - goal_delta[0])
    bottom = tf.minimum(goal_delta[0], traj_len - 1)
    # bottom = tf.minimum(goal_delta[0], traj_len)
    curr_idxs = tf.range(traj_len - bottom)

    # select a random future index for each transition i in the range [i + goal_delta[0], min{all_obs_len, i + goal_delta[1]})
    rand = tf.random.uniform([traj_len - bottom])
    low = tf.cast(curr_idxs + bottom, tf.float32)
    high = tf.cast(tf.minimum(all_obs_len, curr_idxs + goal_delta[1]), tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # very rarely, floating point errors can cause goal_idxs to be out of bounds
    goal_idxs = tf.minimum(goal_idxs, all_obs_len - 1)

    traj_truncated = tf.nest.map_structure(
        lambda x: tf.gather(x, curr_idxs),
        traj,
    )

    # select goals
    traj_truncated["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        all_obs,
    )

    traj_truncated["goal_dists"] = goal_idxs - curr_idxs
    traj_truncated["bottom"] = tf.ones_like(goal_idxs) * bottom
    traj_truncated["traj_len"] = tf.ones_like(goal_idxs) * traj_len

    return traj_truncated


def delta_goals_with_generated(traj, *, goal_delta, frac_generated):
    """
    Relabels with a uniform distribution over future states in the range [i +
    goal_delta[0], min{traj_len, i + goal_delta[1]}). Truncates trajectories to
    have length traj_len - goal_delta[0]. Not suitable for RL (does not add
    terminals or rewards).
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # add the last observation (which only exists in next_observations) to get
    # all the observations
    all_obs = tf.nest.map_structure(
        lambda obs, next_obs: tf.concat([obs, next_obs[-1:]], axis=0),
        traj["observations"],
        traj["next_observations"],
    )
    all_obs_len = traj_len + 1

    # current obs should only come from [0, traj_len - goal_delta[0])
    curr_idxs = tf.range(traj_len - goal_delta[0])

    # select a random future index for each transition i in the range [i + goal_delta[0], min{all_obs_len, i + goal_delta[1]})
    rand = tf.random.uniform([traj_len - goal_delta[0]])
    low = tf.cast(curr_idxs + goal_delta[0], tf.float32)
    high = tf.cast(tf.minimum(all_obs_len, curr_idxs + goal_delta[1]), tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32) # what percentage in the range of low to high to be at 

    
    max_goal_idxs = tf.cast(high, tf.int32) # equivalent to if rand=1
    max_goal_idxs = tf.minimum(max_goal_idxs, all_obs_len - 1)
    max_goal_dists = max_goal_idxs - curr_idxs

    # very rarely, floating point errors can cause goal_idxs to be out of bounds
    goal_idxs = tf.minimum(goal_idxs, all_obs_len - 1)

    traj_truncated = tf.nest.map_structure(
        lambda x: tf.gather(x, curr_idxs),
        traj,
    )

    # select goals
    traj_truncated["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        all_obs,
    )

    traj_truncated["goal_dists"] = goal_idxs - curr_idxs


    # TODO: add if statement so that it only does this if "generated_goals" in traj, since has to handle both lcbc and gcbc data
    # if "generated_goals" in traj:
    if True:
        idxs_of_generated_goals = tf.random.uniform(shape=(tf.shape(traj_truncated["generated_goals"])[0],), minval=0, maxval=tf.shape(traj_truncated["generated_goals"])[1], dtype=tf.int32)

        all_generated_goals = tf.nest.map_structure(
            lambda x: tf.gather(x, idxs_of_generated_goals, axis=1, batch_dims=1),
            traj_truncated["generated_goals"],
        )

        # Shuffle the indices
        shuffled_curr_idxs = tf.random.shuffle(curr_idxs)
        traj_truncated["shuffled_curr_idxs"] = shuffled_curr_idxs
        # Select half of the indices
        # Calculate the number of indices to select
        num_indices_to_select = tf.cast(tf.cast(tf.shape(shuffled_curr_idxs)[0], tf.float32) * frac_generated, tf.int32)
        # num_indices_to_select = int(num_indices_to_select)

        # Select the first half of the shuffled indices
        curr_idxs_with_generated_goals = shuffled_curr_idxs[:num_indices_to_select] 

        selected_generated_goals = tf.gather(all_generated_goals, curr_idxs_with_generated_goals)

        traj_truncated["goals"]["image"] = tf.tensor_scatter_nd_update(traj_truncated["goals"]["image"], tf.expand_dims(curr_idxs_with_generated_goals, axis=-1), selected_generated_goals)

        selected_max_goal_dists = tf.cast(tf.gather(max_goal_dists, curr_idxs_with_generated_goals), tf.int32)
        traj_truncated["goal_dists"] = tf.tensor_scatter_nd_update(traj_truncated["goal_dists"], tf.expand_dims(curr_idxs_with_generated_goals, axis=-1), selected_max_goal_dists)

        traj_truncated["curr_idxs"] = curr_idxs


        traj_truncated["uses_generated_goal"] = tf.zeros_like(curr_idxs)
        traj_truncated["uses_generated_goal"] = tf.tensor_scatter_nd_update(traj_truncated["uses_generated_goal"], tf.expand_dims(curr_idxs_with_generated_goals, axis=-1), tf.ones_like(curr_idxs_with_generated_goals))
    
        traj_truncated["idxs_of_generated_goals"] = idxs_of_generated_goals

        traj_truncated["goals"]["unaugmented_image"] = tf.identity(traj_truncated["goals"]["image"])  

    return traj_truncated


def delta_goals_with_generated_encode_decode(traj, *, goal_delta, frac_generated, frac_encode_decode, frac_noised_encode_decode, zero_goal):
    """
    Relabels with a uniform distribution over future states in the range [i +
    goal_delta[0], min{traj_len, i + goal_delta[1]}). Truncates trajectories to
    have length traj_len - goal_delta[0]. Not suitable for RL (does not add
    terminals or rewards).
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # add the last observation (which only exists in next_observations) to get
    # all the observations
    all_obs = tf.nest.map_structure(
        lambda obs, next_obs: tf.concat([obs, next_obs[-1:]], axis=0),
        traj["observations"],
        traj["next_observations"],
    )
    all_obs_len = traj_len + 1

    # current obs should only come from [0, traj_len - goal_delta[0])
    curr_idxs = tf.range(traj_len - goal_delta[0])

    # select a random future index for each transition i in the range [i + goal_delta[0], min{all_obs_len, i + goal_delta[1]})
    rand = tf.random.uniform([traj_len - goal_delta[0]])
    low = tf.cast(curr_idxs + goal_delta[0], tf.float32)
    high = tf.cast(tf.minimum(all_obs_len, curr_idxs + goal_delta[1]), tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32) # what percentage in the range of low to high to be at 

    
    max_goal_idxs = tf.cast(high, tf.int32) # equivalent to if rand=1
    max_goal_idxs = tf.minimum(max_goal_idxs, all_obs_len - 1)
    max_goal_dists = max_goal_idxs - curr_idxs

    # very rarely, floating point errors can cause goal_idxs to be out of bounds
    goal_idxs = tf.minimum(goal_idxs, all_obs_len - 1)

    traj_truncated = tf.nest.map_structure(
        lambda x: tf.gather(x, curr_idxs),
        traj,
    )

    # select goals
    traj_truncated["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        all_obs,
    )

    traj_truncated["goal_dists"] = goal_idxs - curr_idxs


    # TODO: add if statement so that it only does this if "generated_goals" in traj, since has to handle both lcbc and gcbc data
    # if "generated_goals" in traj:
    if True:

        # Choose which generated goal, which encode_decode goal, and which noised encode_edcode goal to use 
        idxs_of_generated_goals = tf.random.uniform(shape=(tf.shape(traj_truncated["generated_goals"])[0],), minval=0, maxval=tf.shape(traj_truncated["generated_goals"])[1], dtype=tf.int32)
        all_generated_goals = tf.nest.map_structure(
            lambda x: tf.gather(x, idxs_of_generated_goals, axis=1, batch_dims=1),
            traj_truncated["generated_goals"],
        )

        if zero_goal:
            all_generated_goals = tf.zeros_like(all_generated_goals) 

        idxs_of_encode_decode_goals = tf.random.uniform(shape=(tf.shape(traj_truncated["goals"]["encode_decode_image"])[0],), minval=0, maxval=tf.shape(traj_truncated["goals"]["encode_decode_image"])[1], dtype=tf.int32)
        all_encode_decode_goals = tf.nest.map_structure(
            lambda x: tf.gather(x, idxs_of_encode_decode_goals, axis=1, batch_dims=1),
            traj_truncated["goals"]["encode_decode_image"],
        )

        idxs_of_noised_encode_decode_goals = tf.random.uniform(shape=(tf.shape(traj_truncated["goals"]["noised_encode_decode_image"])[0],), minval=0, maxval=tf.shape(traj_truncated["goals"]["noised_encode_decode_image"])[1], dtype=tf.int32)
        all_noised_encode_decode_goals = tf.nest.map_structure(
            lambda x: tf.gather(x, idxs_of_noised_encode_decode_goals, axis=1, batch_dims=1),
            traj_truncated["goals"]["noised_encode_decode_image"],
        )

        # Choose which transitions have generated goals, encode_decode goals, and noised encode decode goals
        # Shuffle the indices
        shuffled_curr_idxs = tf.random.shuffle(curr_idxs)
        traj_truncated["shuffled_curr_idxs"] = shuffled_curr_idxs
        # Calculate the number of indices to select
        num_idx_generated_to_select = tf.cast(tf.cast(tf.shape(shuffled_curr_idxs)[0], tf.float32) * frac_generated, tf.int32)
        num_idx_enc_dec_to_select = tf.cast(tf.cast(tf.shape(shuffled_curr_idxs)[0], tf.float32) * frac_encode_decode, tf.int32)
        num_idx_noised_enc_dec_to_select = tf.cast(tf.cast(tf.shape(shuffled_curr_idxs)[0], tf.float32) * frac_noised_encode_decode, tf.int32)

        curr_idxs_with_generated_goals = shuffled_curr_idxs[:num_idx_generated_to_select] 
        curr_idxs_with_encode_decode_goals = shuffled_curr_idxs[num_idx_generated_to_select:num_idx_generated_to_select + num_idx_enc_dec_to_select] 
        curr_idxs_with_noised_encode_decode_goals = shuffled_curr_idxs[num_idx_generated_to_select + num_idx_enc_dec_to_select:num_idx_generated_to_select + num_idx_enc_dec_to_select + num_idx_noised_enc_dec_to_select] 

        # For the selected indices, set the goal image to be the encode decode goal image or the noised encode decode goal image
        
        # Gather the generated/encode_decode/noised encode_decode goals that match the idxs of the transitions chosen to have these kinds of goals
        selected_generated_goals = tf.gather(all_generated_goals, curr_idxs_with_generated_goals)
        selected_encode_decode_goals = tf.gather(all_encode_decode_goals, curr_idxs_with_encode_decode_goals)
        selected_noised_encode_decode_goals = tf.gather(all_noised_encode_decode_goals, curr_idxs_with_noised_encode_decode_goals)

        # For the selected indices, set the goal image to be generated goal image, encode decode, or noised encode decode
        traj_truncated["goals"]["image"] = tf.tensor_scatter_nd_update(traj_truncated["goals"]["image"], tf.expand_dims(curr_idxs_with_generated_goals, axis=-1), selected_generated_goals)
        traj_truncated["goals"]["image"] = tf.tensor_scatter_nd_update(traj_truncated["goals"]["image"], tf.expand_dims(curr_idxs_with_encode_decode_goals, axis=-1), selected_encode_decode_goals)
        traj_truncated["goals"]["image"] = tf.tensor_scatter_nd_update(traj_truncated["goals"]["image"], tf.expand_dims(curr_idxs_with_noised_encode_decode_goals, axis=-1), selected_noised_encode_decode_goals)

        # Update the goal_dists
        selected_max_goal_dists = tf.cast(tf.gather(max_goal_dists, curr_idxs_with_generated_goals), tf.int32)
        traj_truncated["goal_dists"] = tf.tensor_scatter_nd_update(traj_truncated["goal_dists"], tf.expand_dims(curr_idxs_with_generated_goals, axis=-1), selected_max_goal_dists) 

    return traj_truncated

GOAL_RELABELING_FUNCTIONS = {
    "uniform": uniform,
    "last_state_upweighted": last_state_upweighted,
    "geometric": geometric,
    "delta_goals": delta_goals,
    "delta_goals2": delta_goals2,
    "delta_goals_with_generated": delta_goals_with_generated,
    "delta_goals_with_generated_encode_decode": delta_goals_with_generated_encode_decode,
}
