#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_117_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<117> mm_instantiator;

public:
  UnalignedM_117_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_117_MMInstantiatorWrapper__;

}  // namespace LLMMM
