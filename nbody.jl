### A Pluto.jl notebook ###
# v0.19.2

using Markdown
using InteractiveUtils

# ╔═╡ 94053080-c5c5-11ec-2e39-638d9d7701a9
begin
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ 9f396ce9-295b-4d80-8020-542d859abfda
begin
	using CUDA
	using PlutoUI
	using Plots

	AA = AbstractArray
	AF = AbstractFloat
end;

# ╔═╡ 216c518e-a625-4c1a-9164-a8e47ae56e89
md"""
We will either operate in the meter-kilogram-seconds regime or in the AU-M⊙-day regime. The latter will is favoured for the nicer numbers.

The values of the constants are taken from CODATA/NIST when available.
"""

# ╔═╡ 63afcba6-eb71-4441-98b2-2be887c6fd27
begin
	const MKS = false

	if MKS
		# Standard G in m-kg-s
		const NEWTON_G = 6.674_30e-11
	else
		# Normalized to AU-M⊙-day
		const NEWTON_G = 2.959_122e-4	
	end
	
	const METER_PER_AU = 1.495_978_706_91e11
	const KG_PER_SOLAR = 1.988_55e30
	const SEC_PER_DAY = 86_400.0
end;

# ╔═╡ 0982b961-eef9-433b-806c-2d8c03a1de7c
md"""
## Define body properties
"""

# ╔═╡ bce0259d-03fc-4422-b4ff-c2ab5f513f40
struct CelestialBody
	mass::AF
	position::AA
	velocity::AA

	function CelestialBody(mass::AF, position::AA, velocity::AA)
		if !MKS
			mass = mass / KG_PER_SOLAR
			position = position ./ METER_PER_AU
			velocity = velocity .* (SEC_PER_DAY/METER_PER_AU)
		end

		return new(mass, position, velocity)
	end
end

# ╔═╡ 87b70d89-25b6-41fb-8d1c-788910ad12d5
bodies = Dict(
	:sun => CelestialBody(KG_PER_SOLAR, [0., 0., 0.], [0., 0., 0.]),
	:venus => CelestialBody(4.867_3e24, [0., 1.082_10e11, 0.], [-3.5e4, 0., 0.]),
	:earth => CelestialBody(5.972_2e24, [0., METER_PER_AU, 0.], [-3.0e4, 0., 0.])
)

# ╔═╡ 9519f0ac-508d-4934-bacf-942911916208


# ╔═╡ e3632e83-4e69-4373-82a6-336b535c07bb
md"""
## Define integrator
"""

# ╔═╡ d966399c-b70e-43cc-a2eb-fa3e55e309e3
function euler(position::AA, velocity::AA, acceleration::AA, mass::AF, Δt::AF)
	position = position + velocity * Δt
    velocity = velocity + acceleration * Δt

    return position, velocity, compute_acceleration(position, mass)
end

# ╔═╡ fe4f4cc3-059b-4d6d-882c-bba5267c7915
function verlet(position::AA, velocity::AA, acceleration::AA, mass::AF, Δt::AF)
    position = position + velocity * Δt + acceleration * 0.5Δt^2
    new_acceleration = compute_acceleration(position, mass)
    velocity = velocity + (acceleration + new_acceleration) * 0.5Δt

    return position, velocity, new_acceleration
end

# ╔═╡ d5da5f3f-86a5-4288-8c83-50b65af8ca02
function symplectic_euler(position::AA, velocity::AA, acceleration::AA, mass::AF,  Δt::AF)
	position = position + velocity * Δt
    acceleration = compute_acceleration(position, mass)
    velocity = velocity + acceleration * Δt

    return position, velocity, acceleration

end

# ╔═╡ b3dfb48b-c2f6-478a-bd56-8dabfbc671c7
md"""
## Define root finder for mid ttv transit
"""

# ╔═╡ 812db9b0-1f54-408a-9545-febc08e3d492
begin
	g = (pos::AA, vel::AA) -> pos .* vel
	δg = (pos::AA, vel::AA, acc::AA) -> pos .* acc .+ vel.^2
		
	function newton_ttv(
		position::AA, 
		velocity::AA, 
		acceleration::AA, 
		mass::AA;
		integrator::Function = verlet,
		max_iter = 10_000, 
		tolerance = eps()
	)
	
		Δt = 0.
		fail_tolerance = true

		# TODO: Earth index
		for iter = 1:max_iter
			δt = -g(position, velocity)/δg(position, velocity, acceleration)
			Δt = Δt + δt
			
			position, velocity, acceleration = integrator(
				position, 
				velocity, 
				acceleration, 
				mass, 
				δt
			)

			if abs(position ≤ tolerance)
				position = 0.
				fail_tolerance = false
				break
			end
			
		end

		if fail_tolerance
			print("WARNING: Root finding tolerance not met")
		end
		
		return Δt, position, velocity, acceleration
	end

end

# ╔═╡ b9a91f0f-ad6d-46fa-920a-01e8a1676f53
md"""
## Read properties and com
"""

# ╔═╡ 37f6cf61-56bd-44a9-8baf-eefde18a7f4f
begin
	transform_to_com = (x::AA, m::AA) -> x .- sum(x .* m'; dims=2)/sum(m)

	function read_properties(bodies::Dict{Symbol, CelestialBody})
		N = length(bodies)
		
		pos = zeros(3, N)
		vel = zeros(3, N)
		mass = zeros(N)

		earth_ind = 0
		
		for (ind, (body, prop)) in enumerate(bodies)

			mass[ind] = prop.mass
			pos[:, ind] = prop.position
			vel[:, ind] = prop.velocity

			if body == :earth
				earth_ind = ind
			end
		end

		pos = transform_to_com(pos, mass)
		vel = transform_to_com(vel, mass)

		return pos, vel, mass, earth_ind
	end
	
end

# ╔═╡ d3da79cd-c73a-4276-aa3b-7df06c27e840
begin
	(position, velocity, mass, ind) = read_properties(bodies)
	const EARTH = ind
end;

# ╔═╡ Cell order:
# ╠═94053080-c5c5-11ec-2e39-638d9d7701a9
# ╠═9f396ce9-295b-4d80-8020-542d859abfda
# ╟─216c518e-a625-4c1a-9164-a8e47ae56e89
# ╠═63afcba6-eb71-4441-98b2-2be887c6fd27
# ╟─0982b961-eef9-433b-806c-2d8c03a1de7c
# ╠═bce0259d-03fc-4422-b4ff-c2ab5f513f40
# ╠═87b70d89-25b6-41fb-8d1c-788910ad12d5
# ╠═d3da79cd-c73a-4276-aa3b-7df06c27e840
# ╠═9519f0ac-508d-4934-bacf-942911916208
# ╟─e3632e83-4e69-4373-82a6-336b535c07bb
# ╠═d966399c-b70e-43cc-a2eb-fa3e55e309e3
# ╠═fe4f4cc3-059b-4d6d-882c-bba5267c7915
# ╠═d5da5f3f-86a5-4288-8c83-50b65af8ca02
# ╟─b3dfb48b-c2f6-478a-bd56-8dabfbc671c7
# ╠═812db9b0-1f54-408a-9545-febc08e3d492
# ╟─b9a91f0f-ad6d-46fa-920a-01e8a1676f53
# ╠═37f6cf61-56bd-44a9-8baf-eefde18a7f4f
