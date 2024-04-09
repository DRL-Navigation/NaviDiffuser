import torch, numpy, sys, os, math

from env.envs import make_env, read_yaml
from utils import EnveDoubleMoActorAgent
import numpy as np
from collections import deque

def make_train_data(num_step, use_gae, gamma, lam, reward, done, value, next_value, reward_size):
    discounted_return = np.empty([num_step, reward_size])
    
    # Discounted Return
    if use_gae:
        gae = np.zeros(reward_size)
        for t in range(num_step - 1, -1, -1):
            delta = reward[t] + gamma * \
                next_value[t] * (1 - done[t]) - value[t]
            gae = delta + gamma * lam * (1 - done[t]) * gae

            discounted_return[t] = gae + value[t]

    else:
        running_add = next_value[-1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[t] + gamma * running_add * (1 - done[t])
            discounted_return[t] = running_add

    return discounted_return

def envelope_operator(num_worker, num_step, enve_start, sample_size, preference, target, value, reward_size, g_step):
    
    # [w1, w1, w1, w1, w1, w1, w2, w2, w2, w2, w2, w2...]
    # [s1, s2, s3, u1, u2, u3, s1, s2, s3, u1, u2, u3...]

    # weak envelope calculation
    ofs = num_worker * num_step
    target = np.concatenate(target).reshape(-1, reward_size)
    if g_step > enve_start:
        prod = np.inner(target, preference)
        envemask = prod.transpose().reshape(sample_size, -1, ofs).argmax(axis=1)
        envemask = envemask.reshape(-1) * ofs + np.array(list(range(ofs))*sample_size)
        target = target[envemask]
    # For Actor
    adv = target - value

    return target, adv

def generate_w(num_prefence, reward_size, fixed_w=None):
    if fixed_w is not None:
        # w = np.random.randn(num_prefence-1, reward_size-3)
        # # normalize as a simplex
        # w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence-1, 1)
        # w_append = np.full((num_prefence-1, 3), 0.25)
        # w = np.concatenate((w, w_append), axis=1)
        # return np.concatenate(([fixed_w], w))
        w = np.tile(fixed_w, (num_prefence, 1))
        return w
    else:
        w = np.random.randn(num_prefence, reward_size-3)
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence, 1)
        w_append = np.full((num_prefence, 3), 0.25)
        w = np.concatenate((w, w_append), axis=1)
        return w
    
def renew_w(preferences, dim):
    w = np.random.randn(reward_size-3)
    w = np.abs(w) / np.linalg.norm(w, ord=1, axis=0)
    w_append = np.full(3, 0.25)
    w = np.concatenate((w, w_append), axis=0)
    preferences[dim] = w
    return preferences

if __name__ == "__main__":
    cfg = read_yaml('./cfg/5obs_3ped_baseline.yaml')

    output = cfg['output_path']
    logname = 'train.log'
    if not os.path.exists(output): os.makedirs(output)
    sys.stdout = open(output+logname, 'w+')

    agent = EnveDoubleMoActorAgent(**cfg['Agent'])
    num_worker = cfg['Training']['num_worker']
    sample_size = cfg['Training']['sample_size']
    num_step = cfg['Training']['num_step']
    use_gae = cfg['Training']['use_gae']
    gamma = cfg['Training']['gamma']
    lam = cfg['Training']['lam']
    enve_start = cfg['Training']['enve_start']

    action_list = cfg['discrete_actions']

    input_size = cfg['Agent']['input_size']
    output_size = cfg['Agent']['output_size']
    reward_size = cfg['Agent']['reward_size']
    fixed_w = np.array([4, 2, 0., 0., 1., 1.])
    # fixed_w = np.array([1.00, 0.00, 0.00)
    # fixed_w = np.array([0.00, 1.00, 0.00)
    # fixed_w = np.array([0.00, 0.00, 1.00)
    explore_w = generate_w(num_worker, reward_size, fixed_w)

    global_step = 0

    envs = []
    states = []
    for _ in range(num_worker):
        env = make_env(cfg)
        envs.append(env)
        state = env.reset()
        laser = state[0].reshape(1, -1)
        vector = state[1]
        state_formed = np.concatenate((vector, laser), axis=1)
        states.append(state_formed)
    states = np.stack(states)

    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_moreward = [], [], [], [], [], []
        global_step += (num_worker * num_step)

        for _ in range(num_step):
            actions = agent.get_action(states, explore_w)

            next_states, dones, real_dones, morewards = [], [], [], []
            cnt = 0
            # print(explore_w, flush=True)
            for env, action in zip(envs, actions):
                state, reward, done, info = env.step([action_list[action]])
                laser = state[0].reshape(1, -1)
                vector = state[1]
                state_formed = np.concatenate((vector, laser), axis=1)
                next_states.append(state_formed)
                if info['dones_info'] == 5:
                    done = True
                    real_done = True
                elif info['dones_info'] in [1, 2, 3, 10]:
                    done = True
                    real_done = False
                else:
                    done = False
                    real_done = False
                dones.append(done)
                real_dones.append(real_done)
                moreward = (reward - np.array([[0, 5, 10, 0, 0, 0]])).reshape(-1)
                morewards.append(moreward)
                # resample if done
                # if cnt > 0 and done:
                #     explore_w = renew_w(explore_w, cnt)
                cnt += 1

            next_states = np.stack(next_states)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            morewards = np.stack(morewards)

            total_state.append(states)
            total_next_state.append(next_states)
            total_done.append(dones)
            total_action.append(actions)
            total_moreward.append(morewards)

            states = next_states

            # sample_rall += rewards[sample_env_idx]
            # sample_morall = sample_morall + morewards[sample_env_idx]
            # sample_step += 1
            # if real_dones[sample_env_idx]:
            #     sample_episode += 1
            #     agent.anneal()
            #     writer.add_scalar('data/reward', sample_rall, sample_episode)
            #     writer.add_scalar('data/step', sample_step, sample_episode)
            #     writer.add_scalar('data/score', scores[sample_env_idx], sample_episode)
            #     writer.add_scalar('data/x_pos_reward', sample_morall[0], sample_episode)
            #     writer.add_scalar('data/time_penalty', sample_morall[1], sample_episode)
            #     writer.add_scalar('data/death_penalty', sample_morall[2], sample_episode)
            #     writer.add_scalar('data/coin_reward', sample_morall[3], sample_episode)
            #     writer.add_scalar('data/enemy_reward', sample_morall[4], sample_episode)
            #     writer.add_scalar('data/tempreture', agent.T, sample_episode)
            #     sample_rall = 0
            #     sample_step = 0
            #     sample_morall = 0

        # [w1, w1, w1, w1, w1, w1, w2, w2, w2, w2, w2, w2...]
        # [s1, s2, s3, u1, u2, u3, s1, s2, s3, u1, u2, u3...]
        # expand w batch
        update_w = generate_w(sample_size, reward_size, fixed_w)
        total_update_w = update_w.repeat(num_step*num_worker, axis=0)
        # expand state batch
        # WRONG!!! total_state = total_state * args.sample_size
        total_state = np.stack(total_state).transpose(
            [1, 0, 2, 3]).reshape([-1, 1, 965])
        total_state = np.tile(total_state, (sample_size, 1, 1))
        # expand next_state batch
        # WRONG!!! total_next_state = total_next_state * args.sample_size
        total_next_state = np.stack(total_next_state).transpose(
            [1, 0, 2, 3]).reshape([-1, 1, 965])
        total_next_state = np.tile(total_next_state, (sample_size, 1, 1))
        # calculate utility from reward vectors
        total_moreward = np.array(total_moreward).transpose([1, 0, 2])
        avg_moreward = np.mean(total_moreward, axis=1)
        for env_num in range(avg_moreward.shape[0]):
            for reward_num in range(avg_moreward.shape[1]):
                agent.tb_writer.add_scalar('Env_No{}, Reward_No{}'.format(str(env_num), str(reward_num)), avg_moreward[env_num][reward_num].item(), global_step)
        total_moreward = total_moreward.reshape([-1, reward_size])
        total_moreward = np.tile(total_moreward, (sample_size, 1))
        # total_utility here is defined for debugging purporses. See
        # https://github.com/RunzheYang/MORL/issues/12
        # total_utility = np.sum(total_moreward * total_update_w, axis=-1).reshape([-1])
        # expand action batch
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_action = np.tile(total_action, sample_size)
        # expand done batch
        total_done = np.stack(total_done).transpose().reshape([-1])
        total_done = np.tile(total_done, sample_size)

        value, next_value, policy = agent.forward_transition(
            total_state, total_next_state, total_update_w)

        # logging utput to see how convergent it is.
        # policy = policy.detach()
        # m = F.softmax(policy, dim=-1)
        # recent_prob.append(m.max(1)[0].mean().cpu().numpy())
        # writer.add_scalar(
        #     'data/max_prob',
        #     np.mean(recent_prob),
        #     sample_episode)

        total_target = []
        total_adv = []
        for idw in range(sample_size):
            ofs = num_worker * num_step
            for idx in range(num_worker):
                target = make_train_data(num_step, use_gae, gamma, lam,
                                total_moreward[idx*num_step+idw*ofs : (idx+1)*num_step+idw*ofs],
                                total_done[idx*num_step+idw*ofs: (idx+1)*num_step+idw*ofs],
                                value[idx*num_step+idw*ofs : (idx+1)*num_step+idw*ofs],
                                next_value[idx*num_step+idw*ofs : (idx+1)*num_step+idw*ofs],
                                reward_size)
                total_target.append(target)

        total_target, total_adv = envelope_operator(num_worker, num_step, enve_start, sample_size, 
                                                    update_w, total_target, value, reward_size, global_step)

        agent.train_model(
            total_state,
            total_next_state,
            total_update_w,
            total_target,
            total_action,
            total_adv,
            global_step)

        # adjust learning rate
        # if args.lr_schedule:
        #     new_learing_rate = args.learning_rate - \
        #         (global_step / args.max_step) * args.learning_rate
        #     for param_group in agent.optimizer.param_groups:
        #         param_group['lr'] = new_learing_rate
        #         writer.add_scalar(
        #             'data/lr', new_learing_rate, sample_episode)

        if global_step % (num_worker * num_step * 100) == 0:
            num = str(global_step // (num_worker * num_step * 1000))
            torch.save(agent.model.state_dict(), output + "/" + num + ".pt")

        if global_step % cfg['Training']['update_target_critic'] == 0:
            agent.sync()