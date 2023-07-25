"""Generic training & eval loops."""

import orbax.checkpoint

#if not jax.devices('gpu'):
#  raise RuntimeError('We recommend using a GPU to run this notebook')

checkpoint_dir = "first_checkpoint"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

options = orbax.checkpoint.CheckpointManagerOptions(
    save_interval_steps=2, max_to_keep=2
)
mngr = orbax.checkpoint.CheckpointManager(
    checkpoint_dir,
    orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
    options
)

with open('/export/corpora/representations/SLURP/test_out.pickle', 'rb') as f:
    lattice_config, lattice_params = pickle.load(f)

lattice = lattice_config.build()


def untokenize(hyp_labels):
    return ''.join([GRAPHEME_SYMBOLS[i] for i in hyp_labels])


decoder_params = lattice_params["params"]


def train_and_eval(
    TEST_BATCHES,
    train_batches,
    model,
    step,
    optimizer=optax.chain(optax.clip_by_global_norm(3.0), optax.adam(2e-5)),
    num_steps=200000,
    num_steps_per_eval=1000,
    num_eval_steps=2176
):
    # Initialize the model parameters using a fixed RNG seed. Flax linen Modules
    # need to know the shape and dtype of its input to initialize the parameters,
    # we thus pass it the test batch.

    train_rng = jax.random.PRNGKey(22)
    if step is None:
        params = model.init(jax.random.PRNGKey(0), DEV_BATCH,
                            test=True).unfreeze()
        import pprint
        print(
            f" number of params total : {count_number_params(params) - count_number_params(lattice_params)}"
        )
        params["params"]["lattice"] = lattice_params["params"]
        opt_state = optimizer.init(params)
    else:
        params = model.init(jax.random.PRNGKey(0), DEV_BATCH)
        opt_state = optimizer.init(params)
        params, opt_state = mngr.restore(step, items=[params, opt_state])

    # jax.jit compiles a JAX function to speed up execution.
    # `donate_argnums=(0, 1)` means we know that the input `params` and
    # `opt_state` won't be needed after calling `train_step`, so we donate them
    # and allow JAX to use their memory for storing the output.
    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, rng, batch):
        # Compute the loss value and the gradients.
        def loss_fn(params, rng):
            intent_logits = model.apply(
                params, batch, test=False, rngs={"dropout": rng}
            )
            loss_intent = optax.softmax_cross_entropy_with_integer_labels(
                intent_logits, batch["intent"]
            )
            return jnp.mean(loss_intent)

        next_rng, rng = jax.random.split(rng)
        loss, grads = jax.value_and_grad(loss_fn)(params, rng)

        # Compute the actual updates based on the optimizer state and the gradients.
        updates, opt_state = optimizer.update(grads, opt_state, params)
        # Apply the updates.
        params = optax.apply_updates(params, updates)
        params['params']['lattice'] = decoder_params
        return params, opt_state, next_rng, {
            'loss': loss,
            "grads": optax.global_norm(grads)
        }

    # We are not passing additional arguments to jax.jit, so it can be used
    # directly as a function decorator.
    @jax.jit
    def eval_step(params, batch):
        intents_logits = model.apply(params, batch, test=True)
        test_loss = optax.softmax_cross_entropy_with_integer_labels(
            intents_logits, batch["intent"]
        )
        # Test accuracy.
        intents = jnp.argmax(intents_logits, axis=1)
        #Compute accuracies
        return compute_accuracies(intents, batch, test_loss)

    num_done_steps = 0
    while num_done_steps < num_steps:
        for step in tqdm(range(num_steps_per_eval), ascii=True):
            next_batch = next(train_batches)
            params, opt_state, train_rng, train_metrics = train_step(
                params, opt_state, train_rng, next(train_batches)
            )
        mngr.save(num_done_steps, [params, opt_state])

        eval_metrics = {"intents": [], "loss": []}
        for _ in tqdm(range(num_eval_steps), ascii=True):

            test_batch = next(TEST_BATCHES)
            eval_metrics_step = eval_step(params, test_batch)
            for i in eval_metrics:
                eval_metrics[i].append(eval_metrics_step[i])

        num_done_steps += num_steps_per_eval
        print(f'step {num_done_steps}\ttrain {train_metrics}')

        with open("log_file.txt", "a") as log_file:
            log_file.write(
                f"step {num_done_steps}\ttrain {train_metrics} \t eval loss : {jnp.mean(jnp.array(eval_metrics['loss']))} \t eval_accuracy {jnp.mean(jnp.array(eval_metrics['intents']))}"
            )
            log_file.write("\n")
        for i in eval_metrics:
            print(f" {i} : {jnp.mean(jnp.array(eval_metrics[i]))}")


model = Model()
step = mngr.latest_step()
train_and_eval(TEST_BATCHES, TRAIN_BATCHES, model, step)
