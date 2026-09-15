import torch
import torch.nn.functional as F
from torch import nn


class SingleStepSphericalStudent(nn.Module):
    def __init__(
        self,
        init_weight,
        hidden_dim=384,
        geometry="tangent_residual",
    ):
        super().__init__()

        valid_geometries = {
            "ambient_residual",
            "tangent_residual",
        }

        if geometry not in valid_geometries:
            raise ValueError(
                "geometry must be one of "
                f"{sorted(valid_geometries)}, "
                f"got {geometry!r}"
            )

        self.geometry = geometry

        output_dim, input_dim = init_weight.shape

        self.text_flow_init = nn.Linear(
            input_dim,
            output_dim,
            bias=False,
        )

        with torch.no_grad():
            self.text_flow_init.weight.copy_(
                init_weight
            )

        self.text_flow_init.weight.requires_grad = False

        self.down = nn.Linear(
            output_dim,
            hidden_dim,
        )
        self.up = nn.Linear(
            hidden_dim,
            output_dim,
        )

        nn.init.normal_(
            self.down.weight,
            std=0.02,
        )
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, text_features):
        z0 = F.normalize(
            self.text_flow_init(text_features),
            dim=-1,
        )

        residual = self.up(
            F.gelu(
                self.down(z0)
            )
        )

        if self.geometry == "tangent_residual":
            # Project the residual onto the tangent
            # space at z0.
            residual = residual - (
                residual * z0
            ).sum(
                dim=-1,
                keepdim=True,
            ) * z0

        # Both controls use the same spherical
        # retraction. They differ only in whether
        # the residual is tangent-projected.
        return F.normalize(
            z0 + residual,
            dim=-1,
        )

    def trainable_parameter_count(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
