from __future__ import annotations

import pytest

from miles.utils.workers.types import DeployComponent, DeploySelector


class TestDeploySelectorParsing:
    def test_a_bare_component_selects_every_instance_of_it(self):
        """`trainer` is how a run deploys all of its trainers as one release, as it did before roles were split."""
        selector = DeploySelector.parse("trainer")

        assert selector == DeploySelector(component=DeployComponent.TRAINER, instance=None)
        assert selector.selects(DeployComponent.TRAINER, instance="critic")

    def test_an_instance_selects_only_itself(self):
        """One release per role is what lets the roles of a run be sized and lost independently."""
        selector = DeploySelector.parse("trainer:actor")

        assert selector.selects(DeployComponent.TRAINER, instance="actor")
        assert not selector.selects(DeployComponent.TRAINER, instance="critic")

    def test_the_value_it_prints_is_the_value_it_was_parsed_from(self):
        """It is passed down to the pods and printed in errors, so it has to round trip."""
        assert DeploySelector.parse("trainer:actor").value == "trainer:actor"
        assert DeploySelector.parse("inference").value == "inference"

    def test_a_whole_run_selects_every_component(self):
        """`all` is a selector over components, and it must keep deploying exactly what it always did."""
        selector = DeploySelector.parse("all")

        assert all(
            selector.selects(component) for component in DeployComponent if component is not DeployComponent.ALL
        )
        assert not selector.is_split()

    def test_the_primary_deployment_carries_the_engines_of_the_run(self):
        """An engine deployment adds engines to a run rather than moving the run's own engines out of it."""
        assert DeploySelector.parse("primary").selects(DeployComponent.INFERENCE)

    def test_an_engine_deployment_carries_nothing_of_the_primary(self):
        """A second controller, router or session server would drive the same engines against the first."""
        assert not DeploySelector.parse("inference").selects(DeployComponent.PRIMARY)

    def test_rejects_a_name_that_is_not_a_component(self):
        """The components partition the run, so an unknown name would deploy an undefined subset."""
        with pytest.raises(AssertionError, match="--deploy-component"):
            DeploySelector.parse("rollout")

    def test_rejects_an_instance_of_a_component_a_run_has_one_of(self):
        """Two primaries would be two orchestration scripts driving one run against each other."""
        with pytest.raises(AssertionError, match="--deploy-component"):
            DeploySelector.parse("primary:west")

    def test_rejects_a_separator_that_names_no_instance(self):
        """`trainer:` reads as an instance the user forgot to type, not as every trainer."""
        with pytest.raises(AssertionError, match="--deploy-component"):
            DeploySelector.parse("trainer:")

    def test_rejects_the_selector_for_all_components_as_an_instance_holder(self):
        """`all` is not a component, so there is no instance of it to deploy."""
        with pytest.raises(AssertionError, match="--deploy-component"):
            DeploySelector.parse("all:west")
