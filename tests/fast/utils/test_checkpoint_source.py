from argparse import Namespace
from pathlib import Path

from miles.utils.arguments import capture_requested_checkpoint_source, resolve_checkpoint_source


def _write_checkpoint(directory: Path, *, iteration: int) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest_checkpointed_iteration.txt").write_text(f"{iteration}\n")
    return str(directory)


def _make_args(*, load: str | None, save: str | None, **overrides: object) -> Namespace:
    fields: dict[str, object] = dict(
        load=load,
        save=save,
        ckpt_step=None,
        finetune=False,
        no_load_optim=False,
        no_load_rng=False,
        critic_load=None,
        critic_save=None,
        ref_load=None,
        ref_ckpt_step=None,
        hf_checkpoint="/hf",
        megatron_to_hf_mode="raw",
        start_rollout_id=None,
    )
    fields.update(overrides)
    args = Namespace(**fields)
    capture_requested_checkpoint_source(args)
    return args


class TestTheActorLoadsThisRunsOwnCheckpoints:
    def test_the_save_dir_wins_over_a_pretrain_load_dir_once_this_run_saved(self, tmp_path):
        """--load /pretrain --save /run must resume from /run, not replay the whole run from /pretrain."""
        load = _write_checkpoint(tmp_path / "pretrain", iteration=0)
        save = _write_checkpoint(tmp_path / "run", iteration=50)
        args = _make_args(load=load, save=save)

        resolve_checkpoint_source(args)

        assert args.load == save

    def test_the_load_dir_is_kept_while_this_run_has_saved_nothing(self, tmp_path):
        """Before the first save there is nothing in --save, so the run has to start from --load."""
        load = _write_checkpoint(tmp_path / "pretrain", iteration=7)
        args = _make_args(load=load, save=str(tmp_path / "run"))

        resolve_checkpoint_source(args)

        assert args.load == load

    def test_the_load_dir_is_kept_when_it_is_newer_than_the_save_dir(self, tmp_path):
        """A --save left over from an older, shorter run must not drag the run backwards."""
        load = _write_checkpoint(tmp_path / "pretrain", iteration=90)
        _write_checkpoint(tmp_path / "run", iteration=10)
        args = _make_args(load=load, save=str(tmp_path / "run"))

        resolve_checkpoint_source(args)

        assert args.load == load

    def test_a_run_that_never_saved_anywhere_falls_back_to_the_reference_weights(self, tmp_path):
        """The cold-start path stays as it was: no checkpoint anywhere means finetune from --ref-load."""
        args = _make_args(load=str(tmp_path / "pretrain"), save=str(tmp_path / "run"), ref_load="/ref")

        resolve_checkpoint_source(args)

        assert args.load == "/ref"
        assert (args.finetune, args.no_load_optim, args.no_load_rng) == (True, True, True)
        assert args.start_rollout_id == 0

    def test_resolving_twice_is_idempotent(self, tmp_path):
        """A trainer re-resolves on every load_state, so the second answer must equal the first."""
        load = _write_checkpoint(tmp_path / "pretrain", iteration=0)
        save = _write_checkpoint(tmp_path / "run", iteration=50)
        args = _make_args(load=load, save=save)

        resolve_checkpoint_source(args)
        first = (args.load, args.critic_load, args.critic_save)
        resolve_checkpoint_source(args)

        assert (args.load, args.critic_load, args.critic_save) == first


class TestTheCriticLoadsItsOwnCheckpoints:
    def test_the_critic_resumes_from_its_own_save_dir_not_from_the_actors(self, tmp_path):
        """The critic writes to <save>_critic, so loading the actor checkpoint would restore the wrong model."""
        load = _write_checkpoint(tmp_path / "pretrain", iteration=0)
        save = _write_checkpoint(tmp_path / "run", iteration=50)
        critic_save = _write_checkpoint(tmp_path / "run_critic", iteration=50)
        args = _make_args(load=load, save=save)

        resolve_checkpoint_source(args)

        assert args.critic_save == critic_save
        assert args.critic_load == critic_save

    def test_the_critic_starts_from_the_actor_source_before_it_saved_anything(self, tmp_path):
        """A cold critic has no checkpoint of its own and is still built from the actor's weights."""
        load = _write_checkpoint(tmp_path / "pretrain", iteration=3)
        args = _make_args(load=load, save=str(tmp_path / "run"))

        resolve_checkpoint_source(args)

        assert args.critic_load == load

    def test_an_explicit_critic_load_is_honoured_until_the_critic_saved_something_newer(self, tmp_path):
        """--critic-load names where the critic starts; its own newer checkpoint still wins after a restart."""
        load = _write_checkpoint(tmp_path / "pretrain", iteration=0)
        critic_load = _write_checkpoint(tmp_path / "critic_pretrain", iteration=1)
        args = _make_args(load=load, save=str(tmp_path / "run"), critic_load=critic_load)

        resolve_checkpoint_source(args)
        assert args.critic_load == critic_load

        critic_save = _write_checkpoint(tmp_path / "run_critic", iteration=9)
        resolve_checkpoint_source(args)
        assert args.critic_load == critic_save
