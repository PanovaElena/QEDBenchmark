#pragma once

#include "Constants.h"
#include "FP.h"
#include "ParticleTypes.h"
#include "Vectors.h"
#include "VectorsProxy.h"

#include <cmath>
#include <stddef.h>
#include <string>
#include <vector>
#include <functional>

using namespace std;
using namespace pfc;

namespace pfc {

    typedef FP Real;
    typedef Real MassType;
    typedef Real ChargeType;

    struct ParticleType {
        MassType mass;
        ChargeType charge;
    };
    namespace ParticleInfo
    {
        extern std::vector<ParticleType> typesVector;
        extern const ParticleType* types;
        extern short numTypes;
    };

    template<Dimension dimension>
    class ParticleProxy;

    template<Dimension dimension>
    class Particle {
    public:

        // Types for conforming ParticleInterface
        typedef typename VectorTypeHelper<dimension, Real>::Type PositionType;
        typedef typename VectorTypeHelper<Three, Real>::Type MomentumType;
        typedef Real GammaType;
        typedef Real WeightType;
        typedef ParticleTypes TypeIndexType;
        typedef typename ProxyVectorTypeHelper<dimension, Real>::Type PositionTypeProxy;
        typedef typename ProxyVectorTypeHelper<Three, Real>::Type MomentumTypeProxy;
        typedef reference_wrapper<Real> GammaTypeProxy;
        typedef reference_wrapper<Real> WeightTypeProxy;
        typedef reference_wrapper<ParticleTypes> TypeIndexTypeProxy;

        Particle() :
            weight(1),
            gamma(1.0),
            typeIndex(ParticleTypes::Electron)
        {}

        Particle(const PositionType& position, const MomentumType& momentum,
            WeightType weight = 1, TypeIndexType typeIndex = ParticleTypes::Electron) :
            position(position), weight(weight), typeIndex(typeIndex)
        {
            setMomentum(momentum);
        }

        Particle(ParticleProxy<dimension>& particleProxy)
        {
            this->setPosition(particleProxy.getPosition());
            this->setP(particleProxy.getP());
            this->setWeight(particleProxy.getWeight());
            this->setType(particleProxy.getType());
        }

        Particle(ParticleProxy<dimension> particleProxy)
        {
            this->setPosition(particleProxy.getPosition());
            this->setP(particleProxy.getP());
            this->setWeight(particleProxy.getWeight());
            this->setType(particleProxy.getType());
        }

        // !!!!!
        Particle(const Particle& particle) :
            position(particle.position), p(particle.p), weight(particle.weight),
            gamma(particle.gamma), typeIndex(particle.typeIndex) {}
        forceinline Particle& operator=(const Particle& particle)
        {
            position = particle.position;
            p = particle.p;
            weight = particle.weight;
            gamma = particle.gamma;
            typeIndex = particle.typeIndex;
            return *this;
        }

        //PositionTypeProxy& getProxyPosition() { return PositionTypeProxy(position); } //only advanced users
        forceinline PositionType getPosition() const { return position; }
        forceinline void setPosition(const PositionType& newPosition) { position = newPosition; }

        forceinline MomentumType getMomentum() const
        {
            return p * Constants<MassType>::c() * getMass();
        }

        forceinline void setMomentum(const MomentumType& newMomentum)
        {
            p = newMomentum / (Constants<GammaType>::c() * getMass());
            gamma = sqrt(static_cast<GammaType>(1.0) + p.norm2());
        }

        forceinline void setP(const MomentumType& newP)
        {
            p = newP;
            gamma = sqrt(static_cast<GammaType>(1.0) + p.norm2());
        }
        forceinline MomentumTypeProxy getProxyP() { return MomentumTypeProxy(p); } //only advanced users
        forceinline MomentumType getP() const { return p; }

        forceinline MomentumType getVelocity() const
        {
            return p * (Constants<GammaType>::c() / gamma);
        }

        forceinline void setVelocity(const MomentumType& newVelocity)
        {
            p = newVelocity / sqrt(constants::c * constants::c - newVelocity.norm2());
            gamma = sqrt((FP)1 + p.norm2());
        }

        forceinline GammaTypeProxy getProxyGamma() { return GammaTypeProxy(gamma); } //only advanced users
        forceinline GammaType getGamma() const { return gamma; }

        forceinline MassType getMass() const { return ParticleInfo::types[typeIndex].mass; }

        forceinline ChargeType getCharge() const { return ParticleInfo::types[typeIndex].charge; }

        forceinline WeightTypeProxy getProxyWeight() { return WeightTypeProxy(weight); } //only advanced users
        forceinline WeightType getWeight() const { return weight; }
        forceinline void setWeight(WeightType newWeight) { weight = newWeight; }

        forceinline TypeIndexType getType() const { return typeIndex; }
        forceinline void setType(TypeIndexType newType) { typeIndex = newType; }

        void save(std::ostream& os)
        {
            os.write((char*)this, sizeof(Particle<dimension>));
        }
        void load(std::istream& is)
        {
            is.read((char*)this, sizeof(Particle<dimension>));
        }

        // !!!!!!!!!!!!!!!!!!!!!!
        friend bool operator==(const Particle& p1, const Particle& p2) {
            return p1.position == p2.position && p1.p == p2.p &&
                p1.weight == p2.weight && p1.gamma == p2.gamma;
        }

    private:

        PositionType position;
        MomentumType p;
        WeightType weight;
        GammaType gamma;
        TypeIndexType typeIndex;

        template<Dimension dim>
        friend class ParticleProxy;
    };

    typedef Particle<One> Particle1d;
    typedef Particle<Two> Particle2d;
    typedef Particle<Three> Particle3d;


    template<Dimension dimension>
    class ParticleProxy {
    public:

        // Types for conforming ParticleInterface
        typedef typename VectorTypeHelper<dimension, Real>::Type PositionType;
        typedef typename VectorTypeHelper<Three, Real>::Type MomentumType;
        typedef Real GammaType;
        typedef Real WeightType;
        typedef ParticleTypes TypeIndexType;
        typedef typename ProxyVectorTypeHelper<dimension, Real>::Type PositionTypeProxy;
        typedef typename ProxyVectorTypeHelper<Three, Real>::Type MomentumTypeProxy;
        typedef reference_wrapper<GammaType> GammaTypeProxy;
        typedef reference_wrapper<WeightType> WeightTypeProxy;
        typedef reference_wrapper<TypeIndexType> TypeIndexTypeProxy;

        ParticleProxy(PositionTypeProxy& position, MomentumTypeProxy& p,
            WeightTypeProxy& weight, TypeIndexTypeProxy& typeIndex, GammaTypeProxy& gamma) :
            position(position), weight(weight), typeIndex(typeIndex),
            p(p), gamma(gamma)
        {}

        ParticleProxy(PositionType& position, MomentumType& p,
            WeightType& weight, TypeIndexType& typeIndex, GammaType& gamma) :
            position(position), weight(weight), typeIndex(typeIndex),
            p(p), gamma(gamma)
        {}

        ParticleProxy(Particle<dimension>& particle) :
            position(particle.position), weight(particle.weight), typeIndex(particle.typeIndex),
            p(particle.p), gamma(particle.gamma)
        {}

        ParticleProxy& operator=(Particle<dimension>& particle) {  // !!!!!!!
            this->position = particle.position;
            this->weight = particle.weight;
            this->typeIndex = particle.typeIndex;
            this->p = particle.p;
            this->gamma = particle.gamma;
            return *this;
        }

        forceinline PositionTypeProxy& getProxyPosition() { return position; } //only advanced users
        forceinline PositionType getPosition() { return position.toVector(); }
        forceinline void setPosition(const PositionType& newPosition) { position = newPosition; }

        forceinline MomentumType getMomentum() const
        {
            return p * (Constants<MassType>::c() * getMass());
        }
        forceinline void setMomentum(const MomentumType& newMomentum)
        {
            p = newMomentum / (Constants<GammaType>::c() * getMass());
            gamma.get() = sqrt(static_cast<GammaType>(1.0) + p.norm2());
        }

        forceinline MomentumTypeProxy getProxyP() const { return p; } //only advanced users
        forceinline MomentumType getP() { return p.toVector(); }
        forceinline void setP(const  MomentumType& newP)
        {
            p = newP;
            gamma.get() = sqrt(static_cast<GammaType>(1.0) + p.norm2());
        }

        MomentumType getVelocity() const
        {
            return p * (Constants<GammaType>::c() / gamma.get());
        }
        forceinline void setVelocity(const MomentumType& newVelocity)
        {
            p = newVelocity / sqrt(constants::c * constants::c - newVelocity.norm2());
            gamma.get() = sqrt((FP)1 + p.norm2());
        }

        forceinline GammaType getGamma() const { return gamma.get(); }
        forceinline GammaTypeProxy getProxyGamma() const { return gamma; } //only advanced users

        forceinline MassType getMass() const { return ParticleInfo::types[typeIndex.get()].mass; }

        forceinline ChargeType getCharge() const { return ParticleInfo::types[typeIndex.get()].charge; }

        forceinline WeightTypeProxy getProxyWeight() const { return weight; } //only advanced users
        forceinline WeightType getWeight() const { return weight.get(); }
        forceinline void setWeight(WeightType newWeight) { weight.get() = newWeight; }

        forceinline TypeIndexType getType() const { return typeIndex.get(); }
        forceinline void setType(TypeIndexType newType) { typeIndex.get() = newType; }

    private:

        PositionTypeProxy position;
        MomentumTypeProxy p;
        WeightTypeProxy weight;
        GammaTypeProxy gamma;
        TypeIndexTypeProxy typeIndex;
    };

    typedef ParticleProxy<One> ParticleProxy1d;
    typedef ParticleProxy<Two> ParticleProxy2d;
    typedef ParticleProxy<Three> ParticleProxy3d;

} // namespace pfc