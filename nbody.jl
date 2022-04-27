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
	using BenchmarkTools
	using CUDA
	using PlutoUI
	using Plots

	import LinearAlgebra: norm

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
	const SEC_PER_DAY = 86_400
	const SEC_PER_YEAR = 31_557_600
end;

# ╔═╡ 0982b961-eef9-433b-806c-2d8c03a1de7c
md"""
## Define body properties
"""

# ╔═╡ bce0259d-03fc-4422-b4ff-c2ab5f513f40
struct CelestialBody{T1<:AF, T2<:AA}
	mass::T1
	position::T2
	velocity::T2

	function CelestialBody(mass::AF, position::AA, velocity::AA)
		if !MKS
			mass = mass / KG_PER_SOLAR
			position = position ./ METER_PER_AU
			velocity = velocity .* (SEC_PER_DAY/METER_PER_AU)
		end

		return new{Float64, Vector{Float64}}(mass, position, velocity)
	end
end

# ╔═╡ 87b70d89-25b6-41fb-8d1c-788910ad12d5
bodies = Dict(
	:sun => CelestialBody(KG_PER_SOLAR, [0., 0., 0.], [0., 0., 0.]),
	:venus => CelestialBody(4.867_3e24, [0., 1.082_10e11, 0.], [-3.5e4, 0., 0.]),
	:earth => CelestialBody(5.972_2e24, [0., METER_PER_AU, 0.], [-3.0e4, 0., 0.])
)

# ╔═╡ 65f28245-a1cf-4b94-a7b2-23227fc88ad0
md"""
## Read properties and com
"""

# ╔═╡ 286ba8b0-1bf7-41ad-bac6-837f50f265c0
begin
	function transform_to_com!(x::AA, m::AA)
		x .-= sum(x .* m'; dims=2)/sum(m)
		return nothing
	end

	function read_properties(bodies::Dict{Symbol, CelestialBody{Float64, Vector{Float64}}})
		N = length(bodies)
		
		pos = zeros(3, N)
		vel = zeros(3, N)
		mass = zeros(N)

		earth_ind = 0
		
		@inbounds for (ind, (body, prop)) in enumerate(bodies)

			mass[ind] = prop.mass
			pos[:, ind] = prop.position
			vel[:, ind] = prop.velocity

			if body == :earth
				earth_ind = ind
			end
		end

		transform_to_com!(pos, mass)
		transform_to_com!(vel, mass)

		return pos, vel, mass, earth_ind
	end
	
end

# ╔═╡ d3da79cd-c73a-4276-aa3b-7df06c27e840
begin
	(position, velocity, mass, ind) = read_properties(bodies)
	const EARTH = ind
end;

# ╔═╡ e3632e83-4e69-4373-82a6-336b535c07bb
md"""
## Define integrator
"""

# ╔═╡ 3f58a3b7-03a9-47fc-8179-5405b0cecded
function compute_acceleration!(acceleration::AA, position::AA, mass::AA)
	
	N = length(mass)
	
	for ii = 1:N
		a = zeros(size(mass))
		@inbounds @simd for jj = 1:N		
			if ii ≠ jj
				r = @view(position[:, ii]) - @view(position[:, jj])
				a += -NEWTON_G/norm(r)^3 * @view(mass[jj]) .* r
			end
		end
		@inbounds acceleration[:, ii] = a
	end
end

# ╔═╡ d966399c-b70e-43cc-a2eb-fa3e55e309e3
function euler!(position::AA, velocity::AA, acceleration::AA, mass::AA, Δt::AF)
	position .+=  velocity * Δt
    velocity .+= acceleration * Δt
	compute_acceleration!(acceleration, position, mass)
    
	return position, velocity, acceleration
end

# ╔═╡ fe4f4cc3-059b-4d6d-882c-bba5267c7915
function verlet!(position::AA, velocity::AA, acceleration::AA, mass::AA, Δt::AF)
    position .+= velocity * Δt
	position .+= acceleration * 0.5Δt^2
	
    velocity .+= acceleration * 0.5Δt
    compute_acceleration!(acceleration, position, mass)
	velocity .+= acceleration * 0.5Δt

    return position, velocity, acceleration
end

# ╔═╡ d5da5f3f-86a5-4288-8c83-50b65af8ca02
function symplectic_euler!(position::AA, velocity::AA, acceleration::AA, mass::AA,  Δt::AF)
	position .+= velocity * Δt
    compute_acceleration!(acceleration, position, mass)
    velocity .+= acceleration * Δt

    return position, velocity, acceleration

end

# ╔═╡ b3dfb48b-c2f6-478a-bd56-8dabfbc671c7
md"""
## Define root finder for mid ttv transit
"""

# ╔═╡ 812db9b0-1f54-408a-9545-febc08e3d492
begin
	g(pos::AA, vel::AA) = pos .* vel
	δg(pos::AA, vel::AA, acc::AA) = pos .* acc .+ vel.^2
		
	function newton_ttv!(
		position::AA, 
		velocity::AA, 
		acceleration::AA, 
		mass::AA,
		integrator!::Function;
		max_iter::Int = 500, 
		tolerance::Float64 = eps(Float64)
	)
	
		Δt = 0.
		fail_tolerance = true

		@inbounds for _ = 1:max_iter
			δt = -g(position, velocity)[1, EARTH]/δg(position, velocity, acceleration)[1, EARTH]
			Δt += δt
			
			integrator!(
				position, 
				velocity, 
				acceleration, 
				mass, 
				δt
			)

			if abs(position[1, EARTH] ≤ tolerance)
				position[0, EARTH] = 0.
				fail_tolerance = false
				break
			end
			
		end

		if fail_tolerance
			print("WARNING: Root finding tolerance not met")
		end
		
		return position, velocity, acceleration, Δt
	end

end

# ╔═╡ 3de36c8d-c837-4c17-bc13-24402114aef0
md"""
## Simulation 🙏
"""

# ╔═╡ 02080954-c779-4b95-9475-010257e7a5db
md"""
Inputs should always be in AU-M⊙-days
"""

# ╔═╡ 95b4b2b1-2846-4940-9568-6c93807f74fb
function run_simulation!(
	position::AA, 
	velocity::AA,
	mass::AA;
	num_orbit::Int = 3,
	buffer::Int = 2,
	Δt::AF = 1., 
	integrator!::Function = verlet!,
)

	num_iter = Int(÷(
		(num_orbit + buffer) * SEC_PER_YEAR, 
		Δt * SEC_PER_DAY, 
		RoundUp
	))
	
	if MKS
		Δt *= SEC_PER_DAY
	end

	num_dims, num_bodies = size(position)
	# posₕ = zeros((num_dims, num_bodies, num_iter+1))
	# velₕ, accₕ = zeros((num_dims, num_bodies, 2))
	posₕ, velₕ, accₕ = [zeros((num_dims, num_bodies, num_iter+1)) for _ in 1:3]
	
	transit_index = zeros(num_orbit)
	transitᵢ = 1
	
	# initial conditions
	iterᵢ = 1
	posₕ[:, :, 1] = position
	velₕ[:, :, 1] = velocity
	compute_acceleration!(@view(accₕ[:, :, 1]), position, mass)

	# Setting up next index with current value overwriting in loop
	posₕ[:, :, 2] = position
	velₕ[:, :, 2] = velocity
	accₕ[:, :, 2] = accₕ[:, :, 1]
	
	while transitᵢ ≤ num_orbit
		iterᵢ += 1
		
		# update current iter values and set up next iter values for next loop
		posₕ[:, :, iterᵢ+1], velₕ[:, :, iterᵢ+1], accₕ[:, :, iterᵢ+1] = integrator!(
			@view(posₕ[:, :, iterᵢ]),
			@view(velₕ[:, :, iterᵢ]),
			@view(accₕ[:, :, iterᵢ]),
			mass,
			Δt
		)

		# crossed mid-transit line
		if posₕ[1, EARTH, iterᵢ-1] > 0 && posₕ[1, EARTH, iterᵢ] < 0
			posₕ[:, :, iterᵢ+1], velₕ[:, :, iterᵢ+1], accₕ[:, :, iterᵢ+1], ttv_Δt = newton_ttv!(
				@view(posₕ[:, :, iterᵢ]), 
				@view(velₕ[:, :, iterᵢ]), 
				@view(accₕ[:, :, iterᵢ]),
				mass,
				integrator!
			)

			transit_index[transitᵢ] = iterᵢ
			transitᵢ += 1
	
		end
	end
	
	return (
		@view(posₕ[:, :, 1:iterᵢ]), 
		@view(velₕ[:, :, 1:iterᵢ]), 
		@view(accₕ[:, :, 1:iterᵢ]),
		transit_index
	)
	
end

# ╔═╡ de847e21-91e0-4269-8804-e6a56d94108f
@btime run_simulation!(position, velocity, mass; num_orbit=50)

# ╔═╡ Cell order:
# ╠═94053080-c5c5-11ec-2e39-638d9d7701a9
# ╠═9f396ce9-295b-4d80-8020-542d859abfda
# ╟─216c518e-a625-4c1a-9164-a8e47ae56e89
# ╠═63afcba6-eb71-4441-98b2-2be887c6fd27
# ╟─0982b961-eef9-433b-806c-2d8c03a1de7c
# ╠═bce0259d-03fc-4422-b4ff-c2ab5f513f40
# ╠═87b70d89-25b6-41fb-8d1c-788910ad12d5
# ╟─65f28245-a1cf-4b94-a7b2-23227fc88ad0
# ╠═286ba8b0-1bf7-41ad-bac6-837f50f265c0
# ╠═d3da79cd-c73a-4276-aa3b-7df06c27e840
# ╟─e3632e83-4e69-4373-82a6-336b535c07bb
# ╠═3f58a3b7-03a9-47fc-8179-5405b0cecded
# ╠═d966399c-b70e-43cc-a2eb-fa3e55e309e3
# ╠═fe4f4cc3-059b-4d6d-882c-bba5267c7915
# ╠═d5da5f3f-86a5-4288-8c83-50b65af8ca02
# ╟─b3dfb48b-c2f6-478a-bd56-8dabfbc671c7
# ╠═812db9b0-1f54-408a-9545-febc08e3d492
# ╟─3de36c8d-c837-4c17-bc13-24402114aef0
# ╟─02080954-c779-4b95-9475-010257e7a5db
# ╠═95b4b2b1-2846-4940-9568-6c93807f74fb
# ╠═de847e21-91e0-4269-8804-e6a56d94108f
