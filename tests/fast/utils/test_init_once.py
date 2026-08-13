import pytest

from miles.utils.init_once import InitOnce, InitState


class TestInitOnce:
    def test_a_fresh_component_reports_itself_uninitialized(self):
        """A restarted orchestration script decides how to start out of exactly this answer."""
        assert InitOnce(component="Widget").is_initialized is False

    def test_a_component_still_inside_its_init_is_not_initialized_yet(self):
        """Marking a component initialized before it built anything is what hid a half-built fleet."""
        once = InitOnce(component="Widget")

        once.enter()

        assert once.state is InitState.INITIALIZING
        assert once.is_initialized is False

    def test_a_completed_init_marks_the_component_initialized(self):
        """The take-over path has to see the component the previous script built as built."""
        once = InitOnce(component="Widget")

        with once.guard():
            pass

        assert once.state is InitState.COMPLETE
        assert once.is_initialized is True

    def test_an_init_that_raised_is_reported_as_failed_and_never_as_initialized(self):
        """A controller that died before creating its servers must not be taken over as a live one."""
        once = InitOnce(component="InferenceController")

        with pytest.raises(RuntimeError, match="boom"):
            with once.guard():
                raise RuntimeError("boom")

        assert once.state is InitState.FAILED
        assert once.is_initialized is False

    def test_a_second_init_in_one_process_fails_loudly_and_names_the_component(self):
        """Re-initializing a live system behind the back of whoever drives it is the bug this exists for."""
        once = InitOnce(component="TrainerController(actor)")
        with once.guard():
            pass

        with pytest.raises(AssertionError, match=r"TrainerController\(actor\) is complete"):
            once.enter()

    def test_initializing_a_component_whose_init_failed_is_refused(self):
        """A failed init leaves state nobody can reason about, so the pod has to be replaced instead."""
        once = InitOnce(component="Widget")
        with pytest.raises(RuntimeError):
            with once.guard():
                raise RuntimeError("boom")

        with pytest.raises(AssertionError, match="Widget is failed"):
            once.enter()

    def test_asserting_initialized_refuses_a_component_that_never_ran_init(self):
        """load_state reloads state that init built, so it cannot run on a component without it."""
        with pytest.raises(AssertionError, match="not started, not initialized"):
            InitOnce(component="Widget").assert_initialized()

    def test_asserting_initialized_passes_after_init(self):
        """The resume path runs on exactly this state and must not be blocked by its own guard."""
        once = InitOnce(component="Widget")
        with once.guard():
            pass

        once.assert_initialized()
